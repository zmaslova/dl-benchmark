""" Common utilities for working with processes.
"""
import subprocess


def cmd_execution(args, log):
    """ Run cmd using subprocess
    """
    log.info(f'Executed command: {" ".join(args)}')

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        universal_newlines=True,
        shell=False,
    )
    output = []
    for line in iter(proc.stdout.readline, ""):
        log.info(line.strip("\n"))
        output.append(line)
        if line or proc.poll() is None:
            continue
        break
    outs = proc.communicate()[0]

    if outs:
        log.info(outs.strip("\n"))
        output.append(outs)

    log.info("Command completed with exit code: %d", proc.returncode)

    return proc.returncode, "".join(output)
