// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"

class State_ZeroTorque : public FSMState
{
public:
    State_ZeroTorque(int state, std::string state_string = "ZeroTorque")
    : FSMState(state, state_string) {}

    void enter()
    {
        for(int i(0); i < 12; ++i)
        {
            auto & motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp()  = 0;
            motor.kd()  = 0;
            motor.dq()  = 0;
            motor.tau() = 0;
        }
    }

    void run()
    {
        // Track current position so re-enabling kp never causes a jerk
        for(int i(0); i < 12; ++i)
            lowcmd->msg_.motor_cmd()[i].q() = lowstate->msg_.motor_state()[i].q();
    }
};

REGISTER_FSM(State_ZeroTorque)
