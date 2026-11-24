# Managed tools

Arkai can install, update, and remove a small set of tools under its own
state directory without changing your global `PATH` or replacing separate
installations:

```bash
arkai tools install       # list supported installers
arkai tools install pi
arkai tools install omp
arkai tools install opencode
arkai tools install strata
arkai tools list
arkai tools update pi
arkai tools remove pi
```

Supported managed installs are Pi, OMP, OpenCode, and Strata. Pi, OMP, and
OpenCode are agents; Strata is an inference backend. Node.js and npm are
required for the Pi installer. The automatic OMP installer supports macOS
and Linux. When Homebrew is available, `arkai tools install opencode` also
reminds you that `brew install opencode` installs OpenCode system-wide.
Arkai manages OpenCode on macOS only on Apple silicon; on Intel Macs use
Homebrew or OpenCode's official instructions. Follow each project's
official instructions if Arkai's installer does not support your platform.

On macOS, copying from sandboxed Pi to the system clipboard may not work
in Terminal.app, which does not support OSC 52 clipboard writes. Use a
terminal that supports OSC 52, such as iTerm2, kitty, or Alacritty.

Installs, updates, and removals require confirmation and run through
Arkai's configured `arapuca` sandbox when it is available. Without
`arapuca`, Arkai warns that the operation will run with no sandbox and
executes it directly on the host, with its writable locations pointed at
Arkai's managed directories. See the [managed tools
design](../design/managed-tools.md) for install boundaries and ownership
rules.
