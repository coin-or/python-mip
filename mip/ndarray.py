import logging
import numpy as np

logger = logging.getLogger(__name__)


class LinExprTensor(np.ndarray):
    def __new__(cls, *args, **kwargs):
        obj = super(LinExprTensor, cls).__new__(*args, **kwargs)
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(LinExprTensor, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. LinExprTensor():
        #    obj is None
        #    (we're in the middle of the LinExprTensor.__new__
        #    constructor, and self.info will be set when we return to
        #    LinExprTensor.__new__)
        if obj is None:
            return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be LinExprTensor)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is LinExprTensor
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for fields, because this
        # method sees all creation of default objects - with the
        # LinExprTensor.__new__ constructor, but also with
        # arr.view(LinExprTensor).
        #
        # example:
        # self.field = getattr(obj, 'field', None)
        #
        # We do not need to return anything

    def __le__(self, other):
        return np.less_equal(self, other, dtype=object)

    def __ge__(self, other):
        return np.greater_equal(self, other, dtype=object)

    def __eq__(self, other):
        return np.equal(self, other, dtype=object)
