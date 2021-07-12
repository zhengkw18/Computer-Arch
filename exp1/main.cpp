#include <stdio.h>
#include <string.h>
#include <vector>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <algorithm>
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

const u32 CACHE_SIZE = 128 * 1024;
const u32 CACHE_SIZE_LEN = __builtin_ctz(CACHE_SIZE);
const u32 ADDRESS_LEN = 64;
const u32 FILE_NUM = 4;
const u32 LFU_COUNTER_LEN = 20;
const u32 LFU_SMALL_COUNTER_LEN = 3;
const u32 PROTECTED_LRU_COUNTER_LEN = 10;
const char *filenames[FILE_NUM] = {
    "test_trace/1.trace",
    "test_trace/2.trace",
    "test_trace/3.trace",
    "test_trace/4.trace",
};

const char *outputs[FILE_NUM] = {
    "1.log",
    "2.log",
    "3.log",
    "4.log"};

inline u32 ceil(u32 num, u32 divider)
{
    return (num + divider - 1) / divider;
}

inline void setBit(u8 *bitmap, u32 pos, bool val)
{
    bitmap[pos >> 3] = val ? (bitmap[pos >> 3] | (1u << (pos & 7))) : (bitmap[pos >> 3] & ~(1u << (pos & 7)));
}

inline bool getBit(const u8 *bitmap, u32 pos)
{
    return bitmap[pos >> 3] >> (pos & 7) & 1;
}

inline u64 getBits(const u8 *bitmap, u32 start, u32 len)
{
    u32 head_arr_pos = start >> 3;
    u32 tail_arr_pos = (start + len) >> 3;
    u32 head_bit_pos = start & 7;
    u32 tail_bit_pos = (start + len) & 7;
    if (head_arr_pos == tail_arr_pos)
    {
        return (bitmap[head_arr_pos] >> head_bit_pos) & ((1u << len) - 1);
    }
    u64 res = 0;
    res += bitmap[tail_arr_pos] & ((1u << tail_bit_pos) - 1);
    for (u32 i = tail_arr_pos - 1; i > head_arr_pos; i--)
    {
        res <<= 8;
        res += bitmap[i];
    }
    res <<= 8 - head_bit_pos;
    res += bitmap[head_arr_pos] >> head_bit_pos;
    return res;
    // u64 res = 0;
    // for (int i = 0; i < len; i++)
    // {
    //     res |= ((u64)getBit(bitmap, start + i) << i);
    // }
    // return res;
}

inline void setBits(u8 *bitmap, u32 start, u32 len, u64 data)
{
    u32 head_arr_pos = start >> 3;
    u32 tail_arr_pos = (start + len) >> 3;
    u32 head_bit_pos = start & 7;
    u32 tail_bit_pos = (start + len) & 7;
    if (head_arr_pos == tail_arr_pos)
    {
        bitmap[head_arr_pos] = (bitmap[head_arr_pos] & ((1u << head_bit_pos) - 1)) | (data << head_bit_pos) | ((bitmap[head_arr_pos] >> tail_bit_pos) << tail_bit_pos);
        return;
    }
    bitmap[head_arr_pos] = (bitmap[head_arr_pos] & ((1u << head_bit_pos) - 1)) | ((data & ((1u << (8 - head_bit_pos)) - 1)) << head_bit_pos);
    data >>= (8 - head_bit_pos);
    for (u32 i = head_arr_pos + 1; i < tail_arr_pos; i++)
    {
        bitmap[i] = data & ((1u << 8) - 1);
        data >>= 8;
    }
    bitmap[tail_arr_pos] = ((bitmap[tail_arr_pos] >> tail_bit_pos) << tail_bit_pos) | (data & ((1u << tail_bit_pos) - 1));
    // for (int i = start; i < start + len; i++)
    // {
    //     setBit(bitmap, i, data & 1);
    //     data >>= 1;
    // }
}

enum Replace
{
    LRU,
    TREE,
    LFU,
    LFU_SMALL,
    LFU_SMALL_PROTECT,
    PROTECTED_LRU_1,
    PROTECTED_LRU_2
};

struct Access
{
    u64 addr;
    bool read;
};

struct Line
{
    bool valid, dirty;
    u64 tag;
};

class Cache
{
    u32 num_ways, block_size;
    u32 num_ways_len, offset_len, num_groups, index_len, tag_len;
    u32 line_meta_size, rep_meta_size;
    Replace replace;
    u8 *line_meta, *rep_meta;
    bool write_back, write_alloc;
    u32 protected_way_num, rep_element_len, counter_len;
    Line readLine(const u8 *src)
    {
        Line line;
        u64 meta = getBits(src, 0, write_back + tag_len + 1);
        line.valid = meta & 1;
        line.tag = (meta >> 1) & ((1ull << tag_len) - 1);
        if (write_back)
            line.dirty = (meta >> (1 + tag_len)) & 1;
        return line;
    }

    void writeLine(u8 *dst, Line line)
    {
        u64 meta = write_back ? (u64)line.valid | (line.tag << 1) | ((u64)line.dirty << (1 + tag_len)) : (u64)line.valid | (line.tag << 1);
        setBits(dst, 0, write_back + tag_len + 1, meta);
    }

    u32 search_stack(u8 *rep_base, u32 way)
    {
        for (u32 i = 0; i < num_ways; i++)
        {
            if ((getBits(rep_base, i * rep_element_len, rep_element_len) & ((1ull << num_ways_len) - 1)) == way)
            {
                return i;
            }
        }
        assert(false);
        return -1;
    }

    void push_stack(u8 *rep_base, u32 way)
    {
        for (u32 i = num_ways - 1; i > 0; i--)
        {
            setBits(rep_base, i * rep_element_len, rep_element_len, getBits(rep_base, (i - 1) * rep_element_len, rep_element_len));
        }
        setBits(rep_base, 0, rep_element_len, replace == LRU ? way : way | (1 << num_ways_len));
    }

    u32 flow_up(u8 *rep_base, u32 stack_pos)
    {
        u64 e = getBits(rep_base, stack_pos * rep_element_len, rep_element_len);
        for (u32 i = stack_pos; i > 0; i--)
        {
            setBits(rep_base, i * rep_element_len, rep_element_len, getBits(rep_base, (i - 1) * rep_element_len, rep_element_len));
        }
        setBits(rep_base, 0, rep_element_len, e);
        return e;
    }

    void reverse_path(u8 *rep_base, u32 way)
    {
        u32 current = 1;
        for (int i = num_ways_len - 1; i >= 0; i--)
        {
            bool choice = (way >> i) & 1;
            setBit(rep_base, current, !choice);
            current = (current << 1) + choice;
        }
    }

    u32 follow_tree(u8 *rep_base)
    {
        u32 current = 1;
        for (u32 i = 0; i < num_ways_len; i++)
        {
            current = (current << 1) + getBit(rep_base, current);
        }
        return current - num_ways;
    }

    void lfu_hit(u8 *rep_base, u32 way)
    {
        u64 e = getBits(rep_base, way * rep_element_len, rep_element_len) + 1;
        if (replace == LFU_SMALL_PROTECT && e == (1ull << counter_len))
        {
            for (u32 i = 0; i < num_ways; i++)
            {
                setBits(rep_base, i * rep_element_len, rep_element_len, getBits(rep_base, i * rep_element_len, rep_element_len) >> 1);
            }
            setBits(rep_base, way * rep_element_len, rep_element_len, e >> 1);
        }
        else
        {
            setBits(rep_base, way * rep_element_len, rep_element_len, e);
        }
    }
    u32 find_lfu(u8 *rep_base)
    {
        u32 min = getBits(rep_base, 0, rep_element_len), argmin = 0;
        for (u32 i = 1; i < num_ways; i++)
        {
            u32 v;
            if ((v = getBits(rep_base, i * rep_element_len, rep_element_len)) < min)
            {
                min = v;
                argmin = i;
            }
        }
        setBits(rep_base, argmin * rep_element_len, rep_element_len, 1);
        return argmin;
    }
    u32 search_protected_stack(u8 *rep_base)
    {
        std::vector<u64> freqs;
        for (u32 i = 0; i < num_ways; i++)
        {
            u64 e = getBits(rep_base, i * rep_element_len, rep_element_len);
            u64 freq = e >> num_ways_len;
            freqs.push_back(freq);
        }
        std::sort(freqs.begin(), freqs.end());
        std::reverse(freqs.begin(), freqs.end());
        for (int i = num_ways - 1; i >= 0; i--)
        {
            u64 e = getBits(rep_base, i * rep_element_len, rep_element_len);
            u64 freq = e >> num_ways_len;
            if (freq <= freqs[protected_way_num])
                return i;
        }
        return 0;
    }
    void add_top_counter(u8 *rep_base)
    {
        u64 e = getBits(rep_base, 0, rep_element_len);
        u64 way = e & ((1ull << num_ways_len) - 1), freq = (e >> num_ways_len) + 1;
        if (freq == (1ull << counter_len))
        {
            for (u32 i = 1; i < num_ways; i++)
            {
                u64 e0 = getBits(rep_base, i * rep_element_len, rep_element_len);
                u64 way0 = e0 & ((1ull << num_ways_len) - 1), freq0 = (e0 >> num_ways_len) >> 1;
                setBits(rep_base, i * rep_element_len, rep_element_len, way0 | (freq0 << num_ways_len));
            }
            setBits(rep_base, 0, rep_element_len, way | (freq >> 1 << num_ways_len));
        }
        else
        {
            setBits(rep_base, 0, rep_element_len, way | (freq << num_ways_len));
        }
    }

public:
    Cache(u32 num_ways, u32 block_size, Replace replace, bool write_back, bool write_alloc) : num_ways(num_ways), replace(replace), write_back(write_back), write_alloc(write_alloc)
    {
        num_ways_len = __builtin_ctz(num_ways);
        offset_len = __builtin_ctz(block_size);
        num_groups = CACHE_SIZE / (num_ways * block_size);
        index_len = __builtin_ctz(num_groups);
        tag_len = ADDRESS_LEN - offset_len - index_len;
        line_meta_size = ceil(write_back + tag_len + 1, 8);
        switch (replace)
        {
        case LRU:
            rep_element_len = num_ways_len;
            break;
        case TREE:
            rep_element_len = 1;
            break;
        case LFU:
            rep_element_len = counter_len = LFU_COUNTER_LEN;
            break;
        case LFU_SMALL:
            rep_element_len = counter_len = LFU_SMALL_COUNTER_LEN;
            break;
        case LFU_SMALL_PROTECT:
            rep_element_len = counter_len = LFU_SMALL_COUNTER_LEN;
            break;
        case PROTECTED_LRU_1:
            rep_element_len = num_ways_len + (counter_len = PROTECTED_LRU_COUNTER_LEN);
            protected_way_num = 1;
            break;
        case PROTECTED_LRU_2:
            rep_element_len = num_ways_len + (counter_len = PROTECTED_LRU_COUNTER_LEN);
            protected_way_num = num_ways >> 1;
            break;
        default:
            break;
        }
        rep_meta_size = ceil(num_ways * rep_element_len, 8);
        line_meta = new u8[line_meta_size * num_ways * num_groups];
        rep_meta = new u8[rep_meta_size * num_groups];
        memset(line_meta, 0, line_meta_size * num_ways * num_groups);
        memset(rep_meta, 0, rep_meta_size * num_groups);
    }
    ~Cache()
    {
        delete[] line_meta;
        delete[] rep_meta;
    }
    bool simulate(Access access)
    {
        u32 index = (access.addr >> offset_len) & ((1ull << index_len) - 1);
        u64 tag = (access.addr >> (offset_len + index_len)) & ((1ull << tag_len) - 1);
        u32 line_base = line_meta_size * num_ways * index;
        u8 *rep_base = &rep_meta[rep_meta_size * index];
        int vacant = -1;
        for (u32 i = 0; i < num_ways; i++)
        {
            u8 *line_addr = &line_meta[line_base + i * line_meta_size];
            Line line = readLine(line_addr);
            if (line.valid)
            {
                if (line.tag == tag)
                {
                    // 写回法且写命中时，置dirty位为1
                    if (!access.read && write_back && !line.dirty)
                    {
                        line.dirty = true;
                        writeLine(line_addr, line);
                    }
                    switch (replace)
                    {
                    case LRU:
                        flow_up(rep_base, search_stack(rep_base, i));
                        break;
                    case TREE:
                        reverse_path(rep_base, i);
                        break;
                    case LFU:
                    case LFU_SMALL:
                    case LFU_SMALL_PROTECT:
                        lfu_hit(rep_base, i);
                        break;
                    case PROTECTED_LRU_1:
                    case PROTECTED_LRU_2:
                        flow_up(rep_base, search_stack(rep_base, i));
                        add_top_counter(rep_base);
                        break;
                    default:
                        break;
                    }
                    return true;
                }
            }
            else if (vacant == -1)
            {
                vacant = i;
            }
        }
        if (access.read || write_alloc)
        {
            switch (replace)
            {
            case LRU:
                if (vacant == -1)
                {
                    vacant = flow_up(rep_base, num_ways - 1);
                }
                else
                {
                    push_stack(rep_base, vacant);
                }
                break;
            case TREE:
                // 二叉树的叶子节点被填满时，修改二叉树的valid位为1
                if (vacant == num_ways - 1)
                    *rep_base |= 1;
                if (vacant == -1)
                {
                    vacant = follow_tree(rep_base);
                }
                reverse_path(rep_base, vacant);
                break;
            case LFU:
            case LFU_SMALL:
            case LFU_SMALL_PROTECT:
                if (vacant == -1)
                {
                    vacant = find_lfu(rep_base);
                }
                else
                {
                    setBits(rep_base, vacant * rep_element_len, rep_element_len, 1);
                }
                break;
            case PROTECTED_LRU_1:
            case PROTECTED_LRU_2:
                if (vacant == -1)
                {
                    vacant = flow_up(rep_base, search_protected_stack(rep_base)) & ((1 << num_ways_len) - 1);
                    setBits(rep_base, 0, rep_element_len, vacant | (1 << num_ways_len));
                }
                else
                {
                    push_stack(rep_base, vacant);
                }
                break;
            default:
                break;
            }
            Line line;
            line.valid = true;
            line.tag = tag;
            // 写回法且装入块时，置dirty位为0
            if (write_back)
                line.dirty = false;
            writeLine(&line_meta[line_base + vacant * line_meta_size], line);
        }
        return false;
    }
};

enum Status
{
    EXPECT_0,
    EXPECT_b,
    EXPECT_n
};

std::vector<Access> read_traces(const char *filename)
{
    std::vector<Access> access;
    FILE *fp = fopen(filename, "r");
    int c;
    Status status = EXPECT_0;
    u64 addr = 0;
    while ((c = fgetc(fp)) != EOF)
    {
        switch (status)
        {
        case EXPECT_0:
            if (c == '0')
                status = EXPECT_b;
            break;
        case EXPECT_b:
            if (c == 'b')
                status = EXPECT_n;
            break;
        case EXPECT_n:
            switch (c)
            {
            case '0':
                addr <<= 1;
                break;
            case '1':
                addr <<= 1;
                addr += 1;
                break;
            case 'r':
                access.push_back({addr, true});
                addr = 0;
                status = EXPECT_0;
                break;
            case 'w':
                access.push_back({addr, false});
                addr = 0;
                status = EXPECT_0;
                break;
            default:
                break;
            }
            break;
        default:
            break;
        }
    }
    fclose(fp);
    return access;
}

void test(u32 num_ways, u32 block_size, Replace replace, bool write_back, bool write_alloc, std::vector<Access> &accesses)
{
    Cache cache(num_ways, block_size, replace, write_back, write_alloc);
    u32 hit = 0, miss = 0;
    for (Access access : accesses)
    {
        if (cache.simulate(access))
            hit++;
        else
            miss++;
    }
    printf("Hit: %d Miss: %d Miss ratio: %.2f\n", hit, miss, (float)miss / (hit + miss) * 100);
}
void write_log(u32 num_ways, u32 block_size, Replace replace, bool write_back, bool write_alloc, std::vector<Access> &accesses, const char *output)
{
    Cache cache(num_ways, block_size, replace, write_back, write_alloc);
    FILE *fp = fopen(output, "w");
    for (Access access : accesses)
    {
        fputs(cache.simulate(access) ? "Hit\n" : "Miss\n", fp);
    }
    fclose(fp);
}

int main()
{
    for (int i = 0; i < FILE_NUM; i++)
    {
        std::vector<Access> input = read_traces(filenames[i]);
        printf("Testing file %s\n", filenames[i]);
        for (u32 num_ways : {1, 4, 8, 0})
        {
            for (u32 block_size : {8, 32, 64})
            {
                printf("Testing for %d-ways, block size=%dB\n", num_ways, block_size);
                test(num_ways == 0 ? CACHE_SIZE / block_size : num_ways, block_size, TREE, true, true, input);
            }
        }
        for (Replace replace : {LRU, TREE, LFU, LFU_SMALL, LFU_SMALL_PROTECT, PROTECTED_LRU_1, PROTECTED_LRU_2})
        {
            printf("Testing for replace algorithm %d\n", replace);
            test(8, 8, replace, true, true, input);
        }
        for (bool write_alloc : {true, false})
        {
            printf("Testing for write allocation: %d\n", write_alloc);
            test(8, 8, TREE, true, write_alloc, input);
        }
        write_log(8, 8, TREE, true, true, input, outputs[i]);
    }
    return 0;
}