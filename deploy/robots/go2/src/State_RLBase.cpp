#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unitree/dds_wrapper/robots/go2/go2_sub.h>
#include <unordered_map>

namespace isaaclab
{
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"up",    {1.0f, 0.0f, 0.0f}},
        {"down",  {-1.0f, 0.0f, 0.0f}},
        {"left",  {0.0f, 0.0f, 1.0f}},
        {"right", {0.0f, 0.0f, -1.0f}},
        {"w",     {1.0f, 0.0f, 0.0f}},
        {"s",     {-1.0f, 0.0f, 0.0f}},
        {"a",     {0.0f, 1.0f, 0.0f}},
        {"d",     {0.0f, -1.0f, 0.0f}},
    };

    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    auto it = key_commands.find(key);
    if (it != key_commands.end()) {
        cmd[0] = std::clamp(it->second[0], cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
        cmd[1] = std::clamp(it->second[1], cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
        cmd[2] = std::clamp(it->second[2], cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());
    }
    return cmd;
}
}

class Go2Articulation : public unitree::BaseArticulation<LowState_t::SharedPtr>
{
public:
    Go2Articulation(LowState_t::SharedPtr lowstate)
    : unitree::BaseArticulation<LowState_t::SharedPtr>(lowstate)
    {
        sport_state = std::make_shared<unitree::robot::go2::subscription::SportModeState>();
    }

    void update() override
    {
        unitree::BaseArticulation<LowState_t::SharedPtr>::update();
        // root_quat_w is already computed by base update(); rotate world-frame velocity to body frame
        data.root_lin_vel_b = data.root_quat_w.conjugate() * sport_state->velocity();
    }

    std::shared_ptr<unitree::robot::go2::subscription::SportModeState> sport_state;
};

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<Go2Articulation>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}