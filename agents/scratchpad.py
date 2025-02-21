from .abstract import _ScratchPadHandler, _ScratchPad
import logging

class ScratchPadHandler(_ScratchPadHandler):
    def emit(self, record):
        entry = self.format(record)
        self.records.append(entry)

class ScratchPad(_ScratchPad):
    FMT = "[%(name)s]:  %(message)s"

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = logging.getLogger(self.agent_name)
        self.handler = ScratchPadHandler()
        self.logger.addHandler(self.handler)
        self.handler.setFormatter(logging.Formatter(self.FMT))
        self.logger.setLevel(logging.INFO)

    def __add__(self, log_msg):
        if isinstance(log_msg, str):
            self.logger.info(log_msg)
        else:
            raise NotImplementedError()
    
    def __radd__(self, log_msg):
        if isinstance(log_msg, str):
            self.logger.info(log_msg)
        else:
            raise NotImplementedError()
    
    def __iadd__(self, log_msg):
        self.__add__(log_msg)

        return self

    def __call__(self, *args, **kwds) -> str:
        """
        By default, assume we want to return the full contents of the
        scratchpad formatted to pass
        """
        out = ""
        for msg in self.handler.records:
            out += msg.replace(f"[{self.agent_name}]:  ", "") + "\n"
        
        return out
