#ifndef __TESTS_ALL_IN_ONE_FPS_COUNTER_H
#define __TESTS_ALL_IN_ONE_FPS_COUNTER_H

#include <chrono>
#include <glog/logging.h>
#include <glog/log_severity.h>

class FPSCounter {
public:
    // 构造函数，初始化累加值和开始时间
    FPSCounter() : sum(0), is_running(false) {}

    // 开始计时
    void Start() {
        start_time = std::chrono::high_resolution_clock::now();
        sum = 0;
        is_running = true;
    }

    // 增加帧数计数
    void Count(int i) {
        if (!is_running) {
            LOG(ERROR) << "Please call Start() before counting.";
            return;
        }
        sum += i;
    }

    // 获取 FPS
    double GetFPS() {
        if (!is_running) {
            LOG(ERROR) << "Please call Start() before calculating FPS.";
            return 0.0;
        }

        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = current_time - start_time;
        double duration_seconds = 
                    std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        
        if (duration_seconds == 0) {
            return 0.0; // 避免除以零
        }
        
        return sum / duration_seconds * 1000;
    }

private:
    int sum;  // 累加值
    bool is_running;  // 计时是否运行
    std::chrono::high_resolution_clock::time_point start_time;  // 开始时间
};




#endif