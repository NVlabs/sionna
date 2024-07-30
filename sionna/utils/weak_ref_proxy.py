#
# SPDX-FileCopyrightText: Copyright (c) 2024 ORANGE - Author: Guillaume Larue <guillaume.larue@orange.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""
Base class for WeakRefProxy objects.
"""

import weakref

class WeakRefDict(weakref.WeakValueDictionary):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        #self._data = weakref.WeakValueDictionary(self)
 
    def __getitem__(self, key):
        # Return a weak reference to the object
        weak_ref = weakref.proxy(super().__getitem__(key))
        #weak_ref = weakref.proxy(self._data[key])
        return weak_ref
    
# class WeakRefProxy:
#     """
#     A proxy class that transparently manages weak references to an object.

#     This class allows you to create a weak reference to an object and interact
#     with it as if it were the original object. When the original object is
#     garbage collected, attempts to access its attributes through the proxy
#     will raise a ReferenceError.

#     Attributes:
#         _weakref (weakref.ref): A weak reference to the original object.
#         _class (type): The class of the original object.

#     Methods:
#         __init__(obj):
#             Initializes the proxy with a weak reference to the given object.
#         __getattr__(name):
#             Retrieves the attribute of the referenced object.
#         __setattr__(name, value):
#             Sets the attribute of the referenced object.
#         __call__():
#             Returns the referenced object if it is still alive, otherwise None.
#         __eq__(other):
#             Checks if the proxy is equal to another object.
#         __repr__():
#             Returns the string representation of the proxy.
#         __str__():
#             Returns the string representation of the referenced object.
#         __class__():
#             Returns the class of the referenced object.
#         __len__():
#             Returns the length of the referenced object.
#         __getitem__(key):
#             Retrieves an item from the referenced object.
#         __setitem__(key, value):
#             Sets an item in the referenced object.
#         __delitem__(key):
#             Deletes an item from the referenced object.
#         __iter__():
#             Returns an iterator for the referenced object.
#         __contains__(item):
#             Checks if an item is in the referenced object.
#         __add__(other):
#             Adds the referenced object to another object.
#         __sub__(other):
#             Subtracts another object from the referenced object.
#         __mul__(other):
#             Multiplies the referenced object by another object.
#         __truediv__(other):
#             Divides the referenced object by another object.
#         __floordiv__(other):
#             Floor divides the referenced object by another object.
#         __mod__(other):
#             Computes the modulus of the referenced object with another object.
#         __pow__(other, modulo=None):
#             Raises the referenced object to the power of another object.
#         __radd__(other):
#             Adds another object to the referenced object (reversed).
#         __rsub__(other):
#             Subtracts the referenced object from another object (reversed).
#         __rmul__(other):
#             Multiplies another object by the referenced object (reversed).
#         __rtruediv__(other):
#             Divides another object by the referenced object (reversed).
#         __rfloordiv__(other):
#             Floor divides another object by the referenced object (reversed).
#         __rmod__(other):
#             Computes the modulus of another object with the referenced object (reversed).
#         __rpow__(other):
#             Raises another object to the power of the referenced object (reversed).
#         __iadd__(other):
#             In-place adds another object to the referenced object.
#         __isub__(other):
#             In-place subtracts another object from the referenced object.
#         __imul__(other):
#             In-place multiplies another object by the referenced object.
#         __itruediv__(other):
#             In-place divides another object by the referenced object.
#         __ifloordiv__(other):
#             In-place floor divides another object by the referenced object.
#         __imod__(other):
#             In-place computes the modulus of another object with the referenced object.
#         __ipow__(other):
#             In-place raises the referenced object to the power of another object.
#         __neg__():
#             Negates the referenced object.
#         __pos__():
#             Returns the positive value of the referenced object.
#         __abs__():
#             Returns the absolute value of the referenced object.
#         __invert__():
#             Inverts the referenced object.
#         __complex__():
#             Converts the referenced object to a complex number.
#         __int__():
#             Converts the referenced object to an integer.
#         __float__():
#             Converts the referenced object to a float.
#         __round__(ndigits=None):
#             Rounds the referenced object to a given precision.
#         __index__():
#             Returns the index of the referenced object.
#         __enter__():
#             Enters the runtime context related to the referenced object.
#         __exit__(exc_type, exc_value, traceback):
#             Exits the runtime context related to the referenced object.
#     """

#     def __init__(self, obj):
#         """
#         Initializes the WeakRefProxy with a weak reference to the given object.

#         Args:
#             obj: The object to create a weak reference to.
#         """
#         super().__setattr__('_weakref', weakref.ref(obj))
#         super().__setattr__('_class', obj.__class__)

#     def __getattr__(self, name):
#         """
#         Retrieves the attribute of the referenced object.

#         Args:
#             name (str): The name of the attribute to retrieve.

#         Returns:
#             The value of the attribute.

#         Raises:
#             ReferenceError: If the referenced object has been garbage collected.
#         """
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return getattr(obj, name)

#     def __setattr__(self, name, value):
#         """
#         Sets the attribute of the referenced object.

#         Args:
#             name (str): The name of the attribute to set.
#             value: The value to set the attribute to.

#         Raises:
#             ReferenceError: If the referenced object has been garbage collected.
#         """
#         if name in ("_weakref", "_class"):
#             super().__setattr__(name, value)
#         else:
#             obj = self._weakref()
#             if obj is None:
#                 raise ReferenceError("The referenced object has been deleted")
#             setattr(obj, name, value)

#     def __call__(self):
#         """
#         Returns the referenced object if it is still alive, otherwise None.

#         Returns:
#             The referenced object if it is still alive, otherwise None.
#         """
#         return self._weakref()

#     def __eq__(self, other):
#         """
#         Checks if the proxy is equal to another object.

#         Args:
#             other: The object to compare against.

#         Returns:
#             bool: True if the referenced object is equal to the other object, False otherwise.
#         """
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return obj == other

#     def __repr__(self):
#         """
#         Returns the string representation of the proxy.

#         Returns:
#             str: The string representation of the proxy.
#         """
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return repr(obj)

#     def __str__(self):
#         """
#         Returns the string representation of the referenced object.

#         Returns:
#             str: The string representation of the referenced object.
#         """
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return str(obj)

#     @property
#     def __class__(self):
#         """
#         Returns the class of the referenced object.

#         Returns:
#             type: The class of the referenced object.
#         """
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return obj.__class__

#     def __len__(self):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return len(obj)

#     def __getitem__(self, key):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return obj[key]

#     def __setitem__(self, key, value):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         obj[key] = value

#     def __delitem__(self, key):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         del obj[key]

#     def __iter__(self):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return iter(obj)

#     def __contains__(self, item):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return item in obj

#     def __add__(self, other):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return obj + other

#     def __sub__(self, other):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return obj - other

#     def __mul__(self, other):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return obj * other

#     def __truediv__(self, other):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return obj / other

#     def __floordiv__(self, other):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return obj // other

#     def __mod__(self, other):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return obj % other

#     def __pow__(self, other, modulo=None):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return pow(obj, other, modulo)

#     def __radd__(self, other):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return other + obj

#     def __rsub__(self, other):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return other - obj

#     def __rmul__(self, other):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return other * obj

#     def __rtruediv__(self, other):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return other / obj

#     def __rfloordiv__(self, other):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return other // obj

#     def __rmod__(self, other):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return other % obj

#     def __rpow__(self, other):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return pow(other, obj)

#     def __iadd__(self, other):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         obj += other
#         return self

#     def __isub__(self, other):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         obj -= other
#         return self

#     def __imul__(self, other):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         obj *= other
#         return self

#     def __itruediv__(self, other):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         obj /= other
#         return self

#     def __ifloordiv__(self, other):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         obj //= other
#         return self

#     def __imod__(self, other):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         obj %= other
#         return self

#     def __ipow__(self, other):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         obj **= other
#         return self

#     def __neg__(self):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return -obj

#     def __pos__(self):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return +obj

#     def __abs__(self):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return abs(obj)

#     def __invert__(self):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return ~obj

#     def __complex__(self):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return complex(obj)

#     def __int__(self):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return int(obj)

#     def __float__(self):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return float(obj)

#     def __round__(self, ndigits=None):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return round(obj, ndigits)

#     def __index__(self):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return obj.__index__()

#     def __enter__(self):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return obj.__enter__()

#     def __exit__(self, exc_type, exc_value, traceback):
#         obj = self._weakref()
#         if obj is None:
#             raise ReferenceError("The referenced object has been deleted")
#         return obj.__exit__(exc_type, exc_value, traceback)