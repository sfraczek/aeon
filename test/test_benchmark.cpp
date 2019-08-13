
/*******************************************************************************
* Copyright 2016-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <memory>
#include "gtest/gtest.h"
#include "file_util.hpp"
#include "block_loader_file.hpp"
#include "block_manager.hpp"
#include "provider_factory.hpp"

using namespace std;
using namespace nervana;

TEST(benchmark, cache)
{
    char* manifest_root = getenv("TEST_IMAGENET_ROOT");
    char* manifest_name = getenv("TEST_IMAGENET_MANIFEST");
    char* cache_root    = getenv("TEST_IMAGENET_CACHE");
    char* bsz           = getenv("TEST_IMAGENET_BATCH_SIZE");

    if (!manifest_root)
    {
        cout << "Environment vars TEST_IMAGENET_ROOT not found\n";
    }
    else
    {
        size_t batch_size = 128;
        if (bsz)
        {
            std::istringstream iss(bsz);
            iss >> batch_size;
        }

        std::string manifest_filename = "train-index.csv";
        if (manifest_name)
        {
            std::istringstream iss(manifest_name);
            iss >> manifest_filename;
        }
        string manifest = file_util::path_join(manifest_root, manifest_filename);

        bool     shuffle_manifest = false;
        bool     shuffle_enable   = false;
        float    subset_fraction  = 1.0;
        size_t   block_size       = 5000;
        uint32_t random_seed      = 0;
        uint32_t node_id          = 0;
        uint32_t node_count       = 0;
        auto     m_manifest_file  = make_shared<manifest_file>(manifest,
                                                          shuffle_manifest,
                                                          manifest_root,
                                                          subset_fraction,
                                                          block_size,
                                                          random_seed,
                                                          node_id,
                                                          node_count,
                                                          batch_size);

        auto record_count = m_manifest_file->record_count();
        if (record_count == 0)
        {
            throw std::runtime_error("manifest file is empty");
        }
        std::cout << "Manifest file record count: " << record_count << std::endl;
        std::cout << "Block count: " << record_count / block_size << std::endl;

        std::shared_ptr<block_loader_source> m_block_loader =
            make_shared<block_loader_file>(m_manifest_file, block_size);

        auto manager = make_shared<block_manager>(
            m_block_loader, block_size, cache_root, shuffle_enable, random_seed);

        encoded_record_list* records;
        stopwatch            timer;
        timer.start();
        float  count       = 0;
        size_t iterations  = record_count / block_size;
        float  total_count = 0;
        float  total_time  = 0;
        for (size_t i = 0; i < iterations; i++)
        {
            records = manager->next();
            timer.stop();
            count      = records->size();
            float time = timer.get_microseconds() / 1000000.;
            cout << setprecision(0) << "block id=" << i << ", count=" << static_cast<int>(count)
                 << ", time=" << fixed << setprecision(6) << time << " images/second "
                 << setprecision(2) << count / time << "\n";
            total_count += count;
            total_time += time;
            timer.start();
        }
        cout << setprecision(0) << "total count=" << total_count
             << ", total time=" << setprecision(6) << total_time << ", average images/second "
             << total_count / total_time << endl;
    }
}

/* dummy_block_manager
 *
 * Generates random images.
 *
 */
class dummy_block_manager : public async_manager<encoded_record_list, encoded_record_list>
{
    class generator : virtual async_manager_source<encoded_record_list>
    {
        generator() {}
        virtual ~generator() {}
        virtual encoded_record_list* next()                      = 0;
        virtual size_t               record_count() const        = 0;
        virtual size_t               elements_per_record() const = 0;
        virtual void                 reset()                     = 0;
        virtual void                 suspend_output() {}
    };

public:
    dummy_block_manager(size_t block_size,
                        size_t block_count,
                        size_t record_count,
                        size_t elements_per_record)
        : async_manager<encoded_record_list, encoded_record_list>{make_shared<async_manager_source<
                                                                      encoded_record_list>>(),
                                                                  "dummy_block_manager"}
        , m_current_block_number{0}
        , m_block_size{block_size}
        , m_block_count{block_count}
        , m_record_count{record_count}
        , m_elements_per_record{elements_per_record}
        , m_random_generator{random_device{}()}
    {
    }

    virtual ~dummy_block_manager() { finalize(); }
    encoded_record_list* filler() override
    {
        m_state                    = async_state::wait_for_buffer;
        encoded_record_list* rc    = get_pending_buffer();
        m_state                    = async_state::processing;
        encoded_record_list* input = nullptr;

        rc->clear();

        m_state = async_state::fetching_data;
        input   = m_source->next();
        m_state = async_state::processing;
        if (input == nullptr)
        {
            rc = nullptr;
        }
        else
        {
            input->swap(*rc);
        }

        if (++m_current_block_number == m_block_count)
        {
            m_current_block_number = 0;
            m_source->reset();
        }

        if (rc && rc->size() == 0)
        {
            rc = nullptr;
        }

        m_state = async_state::idle;
        return rc;
    }

    virtual void initialize() override { m_current_block_number = 0; }
    size_t       record_count() const override { return m_block_size; }
    size_t       elements_per_record() const override { return m_elements_per_record; }
private:
    size_t            m_current_block_number;
    size_t            m_block_size;
    size_t            m_block_count;
    size_t            m_record_count;
    size_t            m_elements_per_record;
    std::minstd_rand0 m_random_generator;
    bool              m_enable_shuffle;
};

TEST(benchmark, decode_and_transform)
{
    char* bsz          = getenv("TEST_IMAGENET_BATCH_SIZE");
    char* thread_count = getenv("TEST_IMAGENET_DECODE_THREAD_COUNT");

    size_t batch_size          = 128;
    size_t decode_thread_count = 0;
    size_t block_size          = 5005;
    size_t record_count        = 1281167;
    size_t block_count         = 256;
    size_t elements_per_record;
    if (bsz)
    {
        std::istringstream iss(bsz);
        iss >> batch_size;
    }
    if (thread_count)
    {
        std::istringstream iss(thread_count);
        iss >> decode_thread_count;
    }

    auto manager = make_shared<dummy_block_manager>(
        block_size, block_count, record_count, elements_per_record);

    // Default ceil div to get number of batches
    // m_batch_count_value = (record_count + batch_size - 1) / batch_size;
    // m_batch_mode        = BatchMode::ONCE;
    using nlohmann::json;

    int height;
    int width;

    json image_config = {{"type", "image"},
                         {"height", height},
                         {"width", width},
                         {"channels", 3},
                         {"output_type", "float"},
                         {"channel_major", true},
                         {"bgr_to_rgb", true}};

    json label_config = {{"type", "label"}, {"binary", false}};

    auto aug_config = vector<json>{{{"type", "image"},
                                    {"flip_enable", true},
                                    {"center", false},
                                    {"crop_enable", true},
                                    {"horizontal_distortion", {3. / 4., 4. / 3.}},
                                    {"do_area_scale", true},
                                    {"scale", {0.08, 1.0}},
                                    {"mean", {0.485, 0.456, 0.406}},
                                    {"stddev", {0.229, 0.224, 0.225}},
                                    {"resize_short_size", 0}}};

    json config = {{"manifest_root", ""},
                   {"manifest_filename", ""},
                   {"shuffle_enable", true},
                   {"shuffle_manifest", true},
                   {"batch_size", batch_size},
                   {"iteration_mode", "INFINITE"},
                   {"cache_directory", ""},
                   {"decode_thread_count", 0},
                   {"etl", {image_config, label_config}},
                   {"augmentation", aug_config}};

    auto m_provider = provider_factory::create(config_json);

    unsigned int threads_num =
        decode_thread_count != 0 ? decode_thread_count : std::thread::hardware_concurrency();

    const int decode_size = batch_size * ((threads_num * m_input_multiplier - 1) / batch_size + 1);
    m_batch_iterator      = make_shared<batch_iterator>(manager, decode_size);

    m_decoder = make_shared<batch_decoder>(m_batch_iterator,
                                           decode_size,
                                           decode_thread_count,
                                           false /*pinned*/,
                                           m_provider,
                                           random_seed);

    m_final_stage =
        make_shared<batch_iterator_fbm>(m_decoder, batch_size, m_provider, !batch_major);

    m_output_buffer_ptr = m_final_stage->next();
}

// TODO(sfraczek): move benchmarks from test_loader.cpp and other test files here
