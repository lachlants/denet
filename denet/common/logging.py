#thin wrapper around logging to allow multiple arguments (like print)
import sys
import logging

#verbose log level between debug / info
VERBOSE = 15
denet_logger = None
denet_flush=False


def get_log_msg(*args):
    msg=""
    for arg in args:
        msg += str(arg) + " "
    return msg.rstrip(" ")

def debug(*args):
    if denet_logger is None:
        init()
    denet_logger.debug(get_log_msg(*args))
    if denet_flush:
        sys.stdout.flush()

def info(*args):
    if denet_logger is None:
        init()
    denet_logger.info(get_log_msg(*args))
    if denet_flush:
        sys.stdout.flush()

def verbose(*args):
    if denet_logger is None:
        init()
    denet_logger.log(VERBOSE, get_log_msg(*args))
    if denet_flush:
        sys.stdout.flush()

def warning(*args):
    if denet_logger is None:
        init()
    denet_logger.warning(get_log_msg(*args))
    if denet_flush:
        sys.stdout.flush()

def error(*args):
    if denet_logger is None:
        init()
    denet_logger.error(get_log_msg(*args))
    if denet_flush:
        sys.stdout.flush()

def critical(*args):
    if denet_logger is None:
        init()
    denet_logger.critical(get_log_msg(*args))
    if denet_flush:
        sys.stdout.flush()

def exception(*args):
    if denet_logger is None:
        init()
    denet_logger.exception(get_log_msg(*args))
    if denet_flush:
        sys.stdout.flush()

def setLevel(lvl):
    if str(lvl).upper() == "VERBOSE":
        lvl = VERBOSE
    denet_logger.setLevel(lvl)

#logging arguments for argparse
def add_arguments(parser):
    parser.add_argument("--log-level", default="verbose", help="Log level")

def init(args = None, flush=False):
    
    logging.basicConfig(stream=sys.stdout, format="%(message)s")
    
    global denet_logger, denet_flush
    denet_logger = logging.getLogger("denet")
    setLevel(VERBOSE if args is None else args.log_level.upper())
    
    if flush:
        debug("Logging: enabling flushing")
        denet_flush = True

    info("--------------------------------")
    info("Program Cmdline: " + " ".join(sys.argv))
    info("--------------------------------")
