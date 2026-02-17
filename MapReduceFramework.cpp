#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <iostream>
#include "Barrier/Barrier.h"
#include "MapReduceFramework.h"


#define ALLOCATE_INTERMEDIATE_ERROR "system error: failed to allocate IntermediateVec\n"
#define ALLOCATE_ATOMIC_ERROR "system error: failed to allocate atomic counter\n"
#define FAILED_TO_CREATE_THREAD_ERROR "system error: failed to create thread\n"
#define FULL_PERCENTAGE 100.0f



struct ThreadContext {
    int threadID;
    const MapReduceClient& client;
    IntermediateVec* intermediateVec;
    std::atomic<uint64_t>* intermediate_keys;

    ThreadContext(int threadID, const MapReduceClient& client)
            : threadID(threadID), client(client) {
        intermediateVec = new (std::nothrow) IntermediateVec;
        if (!intermediateVec) {
            fprintf(stderr, ALLOCATE_INTERMEDIATE_ERROR);
            exit(1);
        }
        intermediate_keys = new (std::nothrow) std::atomic<uint64_t>(0);
        if (!intermediate_keys) {
            fprintf(stderr, ALLOCATE_ATOMIC_ERROR);
            exit(1);
        }
    }

};

struct JobContext{
    const MapReduceClient& client;
    const InputVec& inputVec;
    OutputVec& outputVec;
    std::atomic_flag is_joined = ATOMIC_FLAG_INIT;
    std::atomic_flag first_thread = ATOMIC_FLAG_INIT;
    std::mutex join_mutex;
    std::vector<std::thread> threads;
    std::vector<IntermediateVec*> vec_of_intermediate_vectors;
    std::vector<IntermediateVec*> post_shuffle;
    std::atomic<uint64_t> post_shuffle_num_of_vectors{0};
    std::atomic<uint64_t> reduce_index{0};
    std::atomic<uint64_t> total_keys{0};
    std::atomic<uint64_t> keys_done{0};
    std::atomic<uint64_t> intermediate_keys{0};
    stage_t job_stage = UNDEFINED_STAGE;
    std::mutex add_intermediate_vector;
    std::mutex output_vec_access;
    std::mutex change_state;
    std::atomic<uint64_t> inputVec_access{0};
    Barrier barrier;
    std::vector<ThreadContext> thread_contexts;

    JobContext(const MapReduceClient& client,
               const InputVec& inputVec,
               OutputVec& outputVec,
               int multiThreadLevel)
            : client(client),
              inputVec(inputVec),
              outputVec(outputVec),
              barrier(multiThreadLevel){}

};


void emit2 (K2* key, V2* value, void* context){
    auto* tc = static_cast<ThreadContext*>(context);
    tc->intermediateVec->emplace_back(key, value);
    (*(tc->intermediate_keys))++;
}


void emit3 (K3* key, V3* value, void* context){
    auto * jc = static_cast<JobContext*>(context);
    std::unique_lock<std::mutex> lock(jc->output_vec_access);
    jc->outputVec.emplace_back(key, value);
}

void waitForJob(JobHandle job) {
    auto* jc = static_cast<JobContext*>(job);
    std::unique_lock<std::mutex> lock(jc->join_mutex);
    if (jc->is_joined.test_and_set()) {
        return;
    }
    for (auto& thread : jc->threads) {
        thread.join();
    }
}



void getJobState(JobHandle job, JobState* state){
    auto* jc = static_cast<JobContext*>(job);
    std::unique_lock<std::mutex> lock(jc->change_state);
    auto keys_done = (float) jc->keys_done.load();
    auto total_keys = (float) jc->total_keys.load();

    state->stage = jc->job_stage;
    state->percentage = 0.0f;
    if (total_keys != 0){
        state->percentage = (float) FULL_PERCENTAGE * (keys_done / total_keys);
    }
}


void closeJobHandle(JobHandle job) {
    waitForJob(job);
    auto* jc = static_cast<JobContext*>(job);

    for (IntermediateVec* vec : jc->vec_of_intermediate_vectors) {
        delete vec;
    }

    for (IntermediateVec* vec : jc->post_shuffle) {
        delete vec;
    }

    for (ThreadContext& tc : jc->thread_contexts) {
        delete tc.intermediate_keys;
    }

    delete jc;
}



void shuffle(JobContext* jobContext)
{
    std::vector<IntermediateVec*>& vecs = jobContext->vec_of_intermediate_vectors;
    std::vector<std::pair<int, K2*>> max_keys;
    for (size_t i = 0; i < vecs.size(); ++i) {
        if (!vecs[i]->empty()) {
            K2* key = vecs[i]->back().first;
            max_keys.emplace_back(i, key);
        }
    }
    while (!max_keys.empty()) {
        K2* max_key = max_keys[0].second;
        for (const auto& pair : max_keys) {
            if (*max_key < *(pair.second)) {
                max_key = pair.second;
            }
        }
        auto* group = new IntermediateVec;
        jobContext->post_shuffle_num_of_vectors++;
        for (auto vec : vecs) {
            while (!vec->empty()) {
                K2* current_key = vec->back().first;
                if (!(*current_key < *max_key) && !(*max_key < *current_key)) {
                    group->push_back(vec->back());
                    jobContext->keys_done++;
                    vec->pop_back();
                } else {
                    break;
                }
            }
        }
        jobContext->post_shuffle.push_back(group);
        max_keys.clear();
        for (size_t i = 0; i < vecs.size(); ++i) {
            if (!vecs[i]->empty()) {
                K2* key = vecs[i]->back().first;
                max_keys.emplace_back(i, key);
            }
        }
    }
}


void thread_func(ThreadContext* tc, JobContext* jobContext) {

    // map
    if (jobContext->first_thread.test_and_set()){
        jobContext->job_stage = MAP_STAGE;
    }
    while(true){
        uint64_t inputVec_index = jobContext->inputVec_access++;
        if (inputVec_index < jobContext->total_keys){
            const InputPair& inputPair = (jobContext->inputVec)[inputVec_index];
            tc->client.map(inputPair.first, inputPair.second,tc);
            (jobContext->keys_done).fetch_add(1);
        } else {
            break;
        }
    }
    // sort
    std::sort(tc->intermediateVec->begin(), tc->intermediateVec->end(),
              [](const IntermediatePair& a, const IntermediatePair& b) {
                  return *(a.first) < *(b.first);
              });
    if (*(tc->intermediate_keys) > 0) {
        std::unique_lock<std::mutex> lock(jobContext->add_intermediate_vector);
        jobContext->vec_of_intermediate_vectors.push_back(tc->intermediateVec);
        jobContext->intermediate_keys += *(tc->intermediate_keys);
        lock.unlock();
    }
    jobContext->barrier.barrier();

    // shuffle (only thread 0)
    if (tc->threadID == 0){
        std::unique_lock<std::mutex> lock(jobContext->change_state);
        jobContext->keys_done = 0;
        jobContext->total_keys = jobContext->intermediate_keys.load();
        jobContext->job_stage = SHUFFLE_STAGE;
        lock.unlock();
        shuffle(jobContext);
        std::unique_lock<std::mutex> lock1(jobContext->change_state);
        jobContext->job_stage = REDUCE_STAGE;
        jobContext->keys_done = 0;
        jobContext->total_keys = jobContext->post_shuffle_num_of_vectors.load();
        lock1.unlock();
    }
    jobContext->barrier.barrier();
    // reduce
    while(true){
        uint64_t reduce_index = jobContext->reduce_index++;
        if (reduce_index < jobContext->total_keys){
            const IntermediateVec* pairs = (jobContext->post_shuffle)[reduce_index];
            tc->client.reduce(pairs, jobContext);
            (jobContext->keys_done).fetch_add(1);
        } else {
            break;
        }
    }
}

JobHandle startMapReduceJob(const MapReduceClient& client,
                            const InputVec& inputVec,
                            OutputVec& outputVec,
                            int multiThreadLevel)
{
    if (inputVec.empty()) {
        return nullptr;
    }

    auto* jc = new JobContext(client, inputVec, outputVec, multiThreadLevel);
    jc->total_keys = inputVec.size();
    jc->threads.reserve(multiThreadLevel);
    jc->thread_contexts.reserve(multiThreadLevel);

    for (int i = 0; i < multiThreadLevel; ++i) {
        jc->thread_contexts.emplace_back(i, client);
        try {
            jc->threads.emplace_back(thread_func, &jc->thread_contexts[i], jc);
        } catch (const std::system_error& e) {
            fprintf(stderr, FAILED_TO_CREATE_THREAD_ERROR);
            exit(1);
        }

    }
    return jc;
}


