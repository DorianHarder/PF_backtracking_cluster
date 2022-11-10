class Step(object):

    def __init__(self, _slength, s_heading, _height, _current_floor, _current_position,
                 _scale=0.0):
        """
        Python object that stores all relevant information for each step, some of which are needed in the backtracking process
        Parameters
        ----------
        _slength : length of the current step - float
        s_heading : heading of the current step - float
        _height : height of the current step - float
        _current_floor : current floor - integer
        _current_room : current room - Shapely Point-Geometry
        _valid_rooms : currently valid rooms for this step - Shapely MultiPolygon-Geometry
        _current_position : position of the current step - Shapely Point-Geometry
        _scale : correction value for the step length - float
        """
        self.length = _slength + _scale  # scaling for the step length.
        self.heading = s_heading
        # self.height = _height
        self.current_floor = _current_floor
        # self.current_position = _current_position
        self.scale = _scale
        self.height = _height
        self.current_position = _current_position

        return
