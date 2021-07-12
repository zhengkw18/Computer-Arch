import jittor as jt
from jittor import init, Module
import numpy as np
from pool import Pool3D
import json as js

scores = [40, 70, 100]
time_rate = 0.95


class Pool3D_STD(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False, count_include_pad=True, op="maximum"):
        assert dilation == None
        assert return_indices == None
        assert op == "maximum"
        self.kernel_size = kernel_size
        self.op = op
        self.stride = stride if stride else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad and padding != 0

    def execute(self, x):
        N, C, T, H, W = x.shape
        if self.ceil_mode == False:
            t = (T + self.padding * 2 - self.kernel_size) // self.stride + 1
            h = (H + self.padding * 2 - self.kernel_size) // self.stride + 1
            w = (W + self.padding * 2 - self.kernel_size) // self.stride + 1
        else:
            t = (T + self.padding * 2 - self.kernel_size + self.stride - 1) // self.stride + 1
            h = (H + self.padding * 2 - self.kernel_size + self.stride - 1) // self.stride + 1
            w = (W + self.padding * 2 - self.kernel_size + self.stride - 1) // self.stride + 1
        with jt.flag_scope(compile_options={"max_parallel_depth": 8}):
            xx = x.reindex(
                [N, C, t, h, w, self.kernel_size, self.kernel_size, self.kernel_size],
                [
                    "i0",  # Nid
                    "i1",  # Cid
                    f"i2*{self.stride}-{self.padding}+i5",  # Tid
                    f"i3*{self.stride}-{self.padding}+i6",  # Hid
                    f"i4*{self.stride}-{self.padding}+i7",  # Wid
                ],
            )
            return xx.reduce(self.op, [5, 6, 7])


def big_num(n):
    return str(round(float(n) / 1000 / 1000 / 1000, 2)) + "G"


def print_ans(ans):
    for ans_ in ans:
        for ans__ in ans_:
            for ans___ in ans__:
                print(big_num(ans___), end="=")
            print("", end=" | ")
        print("")


def get_score(user_time, std_time, correct):
    out_scores = []
    cnt = 0
    for i in range(len(user_time)):
        for j in range(len(user_time[i])):
            ut = float(user_time[i][j][0])
            st_ = std_time[i][j]
            st = [float(st_[0]), float(st_[1]), float(st_[2])]
            st.sort()
            cnt += 1
            print("{}: {}".format(cnt, ut / st[2]))
            if ut >= st[2] * time_rate:
                out_scores.append(scores[2])
            elif ut >= st[1] * time_rate:
                out_scores.append(scores[1])
            elif ut >= st[0] * time_rate:
                out_scores.append(scores[0])
            else:
                out_scores.append(0)
            out_scores[-1] *= correct[len(out_scores) - 1]
    return out_scores


# pool3d
jt.flags.use_cuda = 1
data = js.load(open("data.json", "r"))
shapes = data["shapes"]
model_params = data["params"]
models = []
for p in model_params:
    models.append(Pool3D(p[0], stride=p[1]))
models_std = []
for p in model_params:
    models_std.append(Pool3D_STD(p[0], stride=p[1]))

cnt = 0
tot = len(models) * len(shapes)
correct = []
for i in range(len(models)):
    m = models[i]
    m_std = models_std[i]
    for j in range(len(shapes)):
        s = shapes[j]
        x = jt.random(s)
        cnt += 1
        print("testing correctness:", cnt, " / ", tot)
        print(s, m_std.kernel_size, m_std.stride)
        y1 = m(x).data
        y2 = m_std(x).data
        if np.allclose(y1, y2):
            correct.append(1)
        else:
            correct.append(0)

# TODO check all correct
# time
with jt.log_capture_scope(
    log_silent=1,
):
    ans = []
    cnt = 0
    tot = len(models) * len(shapes)
    for i in range(len(models)):
        m = models[i]
        ans_ = []
        for j in range(len(shapes)):
            s = shapes[j]
            x = jt.random(s)

            cnt += 1
            print(cnt, " / ", tot)
            jt.profiler.start(5, 10)
            y = m(x)
            y.sync()
            jt.profiler.stop()
            rep = jt.profiler.report()

            ans_.append([rep[1][-2]])
        ans.append(ans_)
print_ans(ans)
print(correct)
scores = get_score(ans, data["ans"], correct)
print(scores)
sum = 0
for s in scores:
    sum += s
print("avg score:", sum / len(scores))
