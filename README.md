# Overview: copain, a friendly framework for scripting retro games, including AI/RL tools

## Summary

copain is a side project that I started for fun and that could also act as a portfolio project. So far, it is a POC.

It open doors on several worlds I'm curious about:
- scripted tool-assisted speedrunning (if not familiar with the hobby, here's a somewhat formal but exhaustive introduction https://www.youtube.com/watch?v=XG_9r038Qoc )
- IA, reinforcement learning.
- programming

## Core ideas

### Apply complex algorithms to Tool Assisted Speedrun

The first idea is that tool-assisted speedrun could benefit from complex algorithm to produce pleasing gaming runs, either superplays (frame-perfect, best possible timers,...) or plays with an esthetic constraint. As for today such plays are often hand-crafted with thousands of re-records that can take hundred of hours to get right for a given play scenario. The high cost of implementing even one scenario leaves little margin to explore the space of possible game states. Advanced automated search could reveal new ways of playing a game that break it in a fun way, that achieve new performance milestones, or that just are beautiful to watch.

### Implement a general-purpose python interface to emulators for TAS scripting

The second idea is that python scripting will unlock creativity, by opening emulators to the rich ecosystem of python computing packages (numpy, pytorch, etc). That means exposing python bindings for the already existing emulators used for TAS. I target specifically emulators that already expose lua scripting interfaces, because:
- an existing lua interface make interfacing with python easier
- the python interface will be easily portable to all emulators that already support a lua interface
- many emulators that have good emulation accuracy (i.e are faithful to the original gaming hardware), already in use in the TAS community, have lua interfaces

That might sound like reinventing the wheel when considering the many works already online to run various gaming environments for RL purpose. It might be, but at a glance I found that those tools are harder to hack, extend or update than it should be, and often directed toward specific applications, like openai environments. I thought implementing general-purpose python interfaces for emulators would be a good learning experience and a good setup for more general-purpose exploration.

## The python interface

So far, the only supported emulator is FCEUX.

Python bindings for the already existing lua API (https://fceux.com/web/help/LuaFunctionsList.html) are reimplemented in python with IPC. The python process starts the emulator and a socket server. The emulator automatically loads a lua script which triggers a socket client. The python scripting loop of the user can then use a driver class that expose various functions that will under the hood use socket communication to pilot the emulator.

Because it should enable better performances the driver currently uses a UNIX socket, which is exclusively available on POSIX platforms.

TODO: A simple TCP socket, available on all platform, would probably be enough.

From a python scripting perspective, the high-level class `CopainRun` enables starting the environment, and execute a scripting loop that can use the API through the driver handler. See e.g the basic example (running random inputs) in `copain/copain/commands/gradius_random_inputs.py`.

`CopainRun` also supports:
- asynchronously starting several instances of FCEUX. Only one instance of FCEUX will be displayed. The GUI of other instances will be forwarded to display emulators (using xvfb) to save system resources.
- concurrently starting an AI training agent for RL purposes.

## Installation

### Prerequisites

You need:
- `lua5.1` with the package `luaposix`
- `FCEUX` ( https://github.com/TASEmulators/fceux )
- A ROM for the games you would like to script. This repository contains examples using the ROM Gradius (US) . To run those scripts, you need to store the ROM in a folder of your choice and configure the script accordingly. A checksum is performed to ensure that the ROM match the files I used to test the scripts.

Optionally:
- to run several FCEUX instances at once, you will need to install the package `xvfb` (likely available through your package manager).
- to run the RL scripts, you would need to set up pytorch and, for GPU computing, an appropriate cuda environment.

### Docker build

A dockerfile is provided to enable an easier setup of the working environment. It embeds a python environment, a lua5.1 environment, and the fceux executable.

It does not contains `copain`, it is yours to install it within the container.

To build the image, execute `docker build . -t copain` from within the root of the repository.

Then run a container with e.g

`docker run --rm -it --gpus all -e "DISPLAY=$DISPLAY" -v "$HOME/.Xauthority:/root/.Xauthority:ro" --net=host -v /path/to/copain/:/project_dir copain`

where

- `-it` enables interactive mode and a tty
- `--rm` will remove the container when exiting
- `--gpus all` requires the nvidia container toolkit, enables using the gpu from within the container
- `-e "DISPLAY=$DISPLAY" -v "$HOME/.Xauthority:/root/.Xauthority:ro" --net=host` enables running gui (and fceux) from within the container
- `-v /path/to/copain/:/project_dir` enables mounting your local repository in `/project_dir` within the container and subsequently install the project from within this directory.

### Installation

CLI options are not implemented yet, so local installation in editable mode is recommended so you can easily edit a script before running it:

`pip install -e .`

### Running a script

Some scripts are available in the folder `copain/copain/commands`. You can edit the header, before starting the script with e.g

`python -m copain.commands.gradius_random_inputs`

## Examples

Three scripts for gradius are available to try:

- `python -m copain.commands.gradius_random_inputs` will start a visible FCEUX instance, start a 1p game, turn on autofire and then perform random directional inputs. If vic viper dies, the game resets.
- `python -m copain.commands.gradius_brute_force` will start a visible FCEUX instance, start a 1p game and initiate a range of savestates, and perform a naïve brute-force exploration of the gamestates using a die-and-retry strategy, using savestates to unlock a fast exploration of savestates, until it reaches the end of the game. At maximum emulation speed, it takes about 10 minutes to see the end boss (TODO: detection of end game not implemented yet causes an infinite loop after end of game)
- `python -m copain.commands.gradius_brute_force`: a naïve attempt at applying q-learning to vic viper controls, reading the gamestate directly from the game RAM. Runs seemingly smoothly, but so far I got no signs of learning. Probably lacks many analysis tools, q-learning tricks, and understanding of the search space, before getting interesting results. The previous brute-force command might help creating a bank of states/transitions and savestates that would kickstart the learning process.

A nice milestone would be managing to perfect-score the game using a combination of those 3 approaches.

## File tree

```
.
├── copain
│   ├── commands
│   │   ├── gradius_brute_force.py  # brute force example
│   │   ├── gradius_q_learning.py  # attempt at q-learning
│   │   └── gradius_random_inputs.py  # random inputs
│   ├── copain_driver.lua  # lua side implementation of the driver
│   ├── copain_driver.py   # python side implementation of the driver
│   ├── __init__.py
│   ├── nn.py  # classes to define the net architectures for q-learning
│   ├── rl.py  # classes that setup the RL framework
│   ├── run.lua  # lua entrypoint
│   ├── run.py   # python high-level classes to start a run and a scripting loop
│   ├── utils.lua
│   ├── utils.py
│   └── VERSION.txt
├── LICENSE
├── MANIFEST.in
├── README.md   # this readme
├── setup.py
└── Dockerfile  # Dockerfile for building a working run environment
```

## Work in progress

See https://github.com/fcharras/copain/projects/1
