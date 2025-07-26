NOTES:
- For the .devcontainers to work (namely the mouunting of x11 related stuff), the UID env variable has to be set before building the container (to allow substitution on the .devcontainer.json). Do this by adding export UID to the .bashrc_profile and or .zprofile .
- ZSH does not work nicely with .devcontainers (due for example to the command docker version --format {{json .}}). To fix this, we've defined specific profiles:
    -   "terminal.integrated.profiles.linux": {
    "zsh": { "path": "/usr/bin/zsh", "args": ["-l"] }, # This one lets us keep zsh for the main interactive terminal
    "bash-auto": { "path": "/usr/bin/bash", "args": ["-l"] }
  },
  "terminal.integrated.automationProfile.linux": "bash-auto"