class DatasetExceptionBase(Exception):
    ''' The base class for all dataset exceptions'''
    pass

class DatasetFileExists(DatasetExceptionBase, FileExistsError):
    ''' The file you want to save already exists. '''
    pass

class DatasetMissingFrame(DatasetExceptionBase):
    ''' The dataset has no frame. '''
    pass

class DatasetInvalidPath(DatasetExceptionBase):
    '''  The path is invalid and cannot be used for saving or loading. '''
    pass