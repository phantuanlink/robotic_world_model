# Setup (run once per terminal session)

ROS Humble's `libddsc.so.0` (iceoryx-enabled) shadows unitree's noshm build causing crashes.
Run in **both** terminals before starting anything:
```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

# Simulator
```bash
cd ~/Documents/EPFL/MA2/IA\ team/unitree_mujoco/simulate/build
```

```bash
./unitree_mujoco -r go2 -s scene_terrain.xml
```

# Controller
```bash
cd ~/Documents/EPFL/MA2/IA\ team/robotic_world_model/deploy/robots/go2/build
```

```bash
./go2_ctrl --network lo
```

Controls: `f` → FixStand, `v` → velocity controller, `w/s` forward/back, `a/d` strafe, `←/→` rotate, `q` → Passive
