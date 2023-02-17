from abc import ABC, abstractmethod


class BaseFactory(ABC):

    @abstractmethod
    def tuningProcess(self):
        pass

    @abstractmethod
    def modelsBuilder(self):
        pass
