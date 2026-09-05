import { spawn } from "child_process";
import { platform } from "os";

/** Start the server as its own process, detached from Obsidian, so it outlives the plugin
 *  and shows up as a window you can close. Takes a script path (.cmd/.bat/.sh/.exe) or a
 *  command line such as `mmdb serve`. */
export function spawnDetached(command: string): void {
  const cmd = command.trim();
  const bare = cmd.replace(/^"(.*)"$/, "$1");
  const isScript = /\.(cmd|bat|exe|ps1|sh)$/i.test(bare);
  if (platform() === "win32") {
    const target = isScript ? `"${bare}"` : `cmd /k ${cmd}`;
    const child = spawn("cmd.exe", ["/c", `start "mmdb serve" /min ${target}`], {
      detached: true,
      stdio: "ignore",
      windowsVerbatimArguments: true,
    });
    child.unref();
    return;
  }
  const child = spawn("/bin/sh", ["-c", isScript ? `"${bare}"` : cmd], { detached: true, stdio: "ignore" });
  child.unref();
}
