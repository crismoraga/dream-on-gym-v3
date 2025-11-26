class ConnectionEvent:
    """ConnectionEvent
        It is a temporary object that is used in the simulation and stores transmission, transmitter and receiver data.
    """

    def __init__(self):
        """The constructor
            Default values:
            Source : None
            Destination: None
            BitRate: None
            IdConnection: None
        """
        self.__source = None
        self.__destination = None
        self.__bitRate = None
        self.__idConnection = None
    
    ''' Source getter & setter '''
    @property
    def source(self):
        return self.__source
    
    @source.setter
    def source(self,value):
        self.__source = value

    ''' Destination getter & setter '''
    @property
    def destination(self):
        return self.__destination
    
    @destination.setter
    def destination(self,value):
        self.__destination = value

    ''' BitRate getter & setter '''
    @property
    def bitRate(self):
        return self.__bitRate
    
    @bitRate.setter
    def bitRate(self,value):
        self.__bitRate = value
    
    ''' IdConnection getter & setter '''
    @property
    def idConnection(self):
        return self.__idConnection
    
    @idConnection.setter
    def idConnection(self,value):
        self.__idConnection = value