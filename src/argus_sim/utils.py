"""Utility classes and functions."""


class dotdict(dict):
    """A `dict` whose keys can also be read, set and deleted as attributes.

    Examples
    --------
    >>> d = dotdict(a=1, b=2)
    >>> d.a
    1
    >>> d.b = 3
    >>> d['b']
    3
    >>> del d.a
    >>> 'a' in d
    False

    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __dir__(self) -> list:
        """List the keys of the dictionary as attributes.

        Returns
        -------
        list
            A list of keys in the dictionary.

        """
        return list(self.keys())
