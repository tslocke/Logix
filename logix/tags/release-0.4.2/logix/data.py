# {{{ GPL License
# Logix - an extensible programming language for Python
# Copyright (C) 2004 Tom Locke
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# (http://www.gnu.org/copyleft/gpl.html)
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
# }}}

# {{{ imports
import itertools as itt
import operator

try:
    from livedesk.util import debug, debugmode, dp
except ImportError: pass
# }}}

# {{{ Meta
class Meta:

    def __repr__(self):
        return "<Meta " + ' '.join(['%s=%s' % (n,v)
                                    for n,v in self.__dict__.items()]) + ">"

def setmeta(obj, **kws):
    meta = getattr(obj, '__meta__', None)
    if not meta:
        meta = Meta()
        obj.__meta__ = meta
    vars(meta).update(kws)

def getmeta(obj, fieldname):
    meta = getattr(obj, '__meta__', None)
    return meta and getattr(meta, fieldname, None)

def copymeta(src, dest):
    meta = getattr(src, '__meta__', None)
    if meta:
        copy = Meta()
        copy.__dict__ = meta.__dict__.copy()
        dest.__meta__ = copy
# }}}

# {{{ def pprint(head, elems, fields, indent, ...):
def pprint(head, elems, fields, indent,
           lparen="(", rparen=")", separator=''):
    def pprn(ob, indent):
        if hasattr(ob.__class__, '__pprint__'):
            return ob.__pprint__(indent+1, parens=(separator == ""))
        else:
            return repr(ob)

    if head:
        newindent = indent + len(head) + len(lparen) + 1
    else:
        newindent = indent + len(lparen)

    elemstrs = [pprn(e, newindent) for e in elems]

    fieldstrs = ["%s=%s" % (name, pprn(val, newindent + len(name) + 1))
                 for name, val in fields]

    linelen = 0
    all = elemstrs + fieldstrs
    for x in all:
        if x.count("\n") > 0 or len(x) + newindent > 80:
            multiline = True
            break
        linelen += len(x.strip()) + 1
        if linelen > 100:
            multiline = True
            break
    else:
        multiline = False

    if multiline:
        indentstr = ' ' * newindent
        headstr = head and head + " " or ""
        s = (separator + '\n').join([headstr + all[0]] + [indentstr + x
                                                          for x in all[1:]])
    else:
        s = (separator + ' ').join(head and [head] + all or all)

    return lparen + s + rparen
# }}}

# {{{ class flist - a list with an ordered set of named fields
class flist(object):

    # {{{ def new(clss, elems, fields):
    def new(clss, args, kws):
        ml = flist()
        ml.elems = list(args)
        ml.fields = kws.copy()
        return ml
    new = classmethod(new)
    # }}}

    # {{{ def __init__(self, val):
    def __init__(self, *args, **kws):
        if len(args) == 1 and len(kws) == 0:
            # {{{ convert single argument to flist
            val = args[0]
            if val is None:
                self.elems = []
                self.fields = {}

            elif isinstance(val, list):
                self.elems = val[:]
                self.fields = {}

            elif isinstance(val, tuple):
                self.elems = list(val)
                self.fields = {}

            elif isinstance(val, dict):
                self.fields = val.copy()
                self.elems = []

            else:
                raise TypeError("cannot create flist from %r" % val)
            # }}}

        else:
            self.elems = list(args)
            self.fields = kws
    # }}}

    # {{{ def __getitem__(self, key):
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.fields[key]
        else:
            return self.elems[key]
    # }}}
        
    # {{{ def __setitem__(self, key, val):
    def __setitem__(self, key, val):
        if isinstance(key, str):
            self.fields[key] = val
        else:
            self.elems[key] = val
    # }}}

    # {{{ def __iter__(self):
    def __iter__(self):
        for x in self.elems:
            yield x
        for x in self.fields.values():
            yield x
    # }}}

    # {{{ def __repr__(self):
    def __repr__(self):
        return self.__pprint__(0)
    # }}}

    # {{{ def __pprint__(self, indent):
    def __pprint__(self, indent, parens=False):
        return pprint(None, self.elems, self.items(), indent,
                      lparen='[', rparen=']', separator=',')
    # }}}

    # {{{ def __nonzero__(self):
    def __nonzero__(self):
        return bool(self.elems) or bool(self.fields)
    # }}}

    # {{{ def get(self, key, default=None):
    def get(self, key, default=None):
        return self.fields.get(key, default)
    # }}}

    # {{{ def hasField(self, name):
    def hasField(self, name):
        return name in self.fields
    # }}}

    # {{{ def extend(self, other):
    def extend(self, other):
        if isinstance(other, flist):
            self.elems.extend(other.elems)
            self.fields.update(other.fields)
        else:
            self.elems.extend(list(other))
    # }}}

    # {{{ def items(self):
    def items(self):
        return self.fields.items()
    # }}}

    # {{{ def copy(self):
    def copy(self):
        return flist.new(self.elems, self.fields)
    # }}}

    # {{{ def append(self, qx):
    def append(self, x):
        self.elems.append(x)
    # }}}

    # {{{ def __add__(self, other):
    def __add__(self, other):
        new = self.copy()
        new.extend(other)
        return new
    # }}}

    # {{{ def __radd__(self, other):
    def __radd__(self, other):
        return flist.new(list(other) + self.elems, self.fields)
    # }}}

    # {{{ def __eq__(self, other):
    def __eq__(self, other):
        return (isinstance(other, flist)
                and self.elems == other.elems
                and self.fields == other.fields)
    # }}}

    # {{{ def __ne__(self, other):
    def __ne__(self, other):
        return ((not isinstance(other, flist))
                or self.elems != other.elems
                or self.fields != other.fields)
    # }}}

    # {{{ def __len__(self):
    def __len__(self):
        return len(self.elems) + len(self.fields)
    # }}}
# }}}

# {{{ OLD class flist - a list with an ordered set of named fields
##class flist(object):
##
#    # {{{ def new(clss, elems, fields):
#    def new(clss, elems, fields):
#        ml = flist()
#        
#        ml.elems = elems[:]
#        ml.fields = fields.copy()
#
#        fieldOrder = []
#        for n,v in fields:
#            if n in fieldOrder:
#                raise ValueError("duplicate field name in flist %r" % n)
#            else:
#                fieldOrder.append(n)
#        ml._fieldOrder = fieldOrder
#        ml.fields = dict(fields)
#        
#        return ml
#    new = classmethod(new)
#    # }}}
##
##    # {{{ def __init__(self, val):
#    def __init__(self, val=None):
#        if val is None:
#            self.elems = []
#            self.fields = {}
#        
#        elif isinstance(val, list):
#            self.elems = val[:]
#            self.fields = {}
#            
#        elif isinstance(val, tuple):
#            self.elems = list(val)
#            self.fields = {}
#            
#        elif isinstance(val, dict):
#            self.fields = dict.copy()
#            self.elems = []
#
#        else:
#            raise TypeError("cannot create flist from %r" % val)
#            
#        self._fieldOrder = []
#    # }}}
##
##    # {{{ def __getitem__(self, key):
#    def __getitem__(self, key):
#        if isinstance(key, str):
#            return self.fields[key]
#        else:
#            return self.elems[key]
#    # }}}
##        
##    # {{{ def __setitem__(self, key, val):
#    def __setitem__(self, key, val):
#        if isinstance(key, str):
#            self.fields[key] = val
#            if key in self._fieldOrder:
#                self._fieldOrder.remove(key)
#            self._fieldOrder.append(key)
#        else:
#            self.elems[key] = val
#    # }}}
##
##    # {{{ def __iter__(self):
#    def __iter__(self):
#        for x in self.elems:
#            yield x
#        for n in self._fieldOrder:
#            yield self.fields[n]
#    # }}}
##
##    # {{{ def __repr__(self):
#    def __repr__(self):
#        return self.__pprint__(0)
#    # }}}
##
##    # {{{ def __pprint__(self, indent):
#    def __pprint__(self, indent, parens=False):
#        return pprint(None, self.elems, self.items(), indent,
#                      lparen='[', rparen=']', separator=',')
#    # }}}
##
##    # {{{ def __nonzero__(self):
#    def __nonzero__(self):
#        return bool(self.elems) or bool(self.fields)
#    # }}}
##
##    # {{{ def get(self, key, default=None):
#    def get(self, key, default=None):
#        return self.fields.get(key, default)
#    # }}}
##
##    # {{{ def hasField(self, name):
#    def hasField(self, name):
#        return name in self.fields
#    # }}}
##
##    # {{{ def extend(self, other):
#    def extend(self, other):
#        if isinstance(other, flist):
#            self.elems.extend(other.elems)
#            for n in other._fieldOrder:
#                self[n] = other[n]
#        else:
#            self.elems.extend(list(other))
#    # }}}
##
##    # {{{ def items(self):
#    def items(self):
#        return [(n, self[n]) for n in self._fieldOrder]
#    # }}}
##
##    # {{{ def copy(self):
#    def copy(self):
#        return flist.new(self.elems, self.items())
#    # }}}
##
##    # {{{ def append(self, qx):
#    def append(self, x):
#        self.elems.append(x)
#    # }}}
##
##    # {{{ def __add__(self, other):
#    def __add__(self, other):
#        new = self.copy()
#        new.extend(other)
#        return new
#    # }}}
##
##    # {{{ def __radd__(self, other):
#    def __radd__(self, other):
#        return flist.new(list(other) + self.elems, self.items())
#    # }}}
##
##    # {{{ def __eq__(self, other):
#    def __eq__(self, other):
#        return (isinstance(other, flist)
#                and self.elems == other.elems
#                and self.fields == other.fields)
#    # }}}
##
##    # {{{ def __ne__(self, other):
#    def __ne__(self, other):
#        return ((not isinstance(other, flist))
#                or self.elems != other.elems
#                or self.fields != other.fields)
#    # }}}
##
##    # {{{ def __len__(self):
#    def __len__(self):
#        return len(self.elems) + len(self.fields)
#    # }}}
# }}}

# {{{ class ltup(tuple): DISABLED
## class ltup(tuple):

##     def __new__(cls, *args, **kw):
##         return tuple.__new__(cls, args)

##     def __init__(self, *args, **kw):
##         tuple.__init__(args)
##         self.__keywords__ = kw.copy()

##     def __pprint__(self, indent):
##         args = tuple(self)
##         return pprint(None, args, self.__keywords__, indent,
##                       separator=',', lparen="(,", rparen=")")

##     def __repr__(self):
##         return self.__pprint__(0)

##     def __nonzero__(self):
##         if len(self) > 0:
##             return True

##         return bool(self.__keywords__)

##     def __eq__(self, other):
##         return (isinstance(other, ltup) and
##                 tuple.__eq__(self, other) and
##                 self.__keywords__ == other.__keywords__)

##     def __ne__(self, other):
##         return ((not isinstance(other, ltup)) or
##                 tuple.__ne__(self, other) or
##                 self.__keywords__ != other.__keywords__)

##     def __getattr__(self, name):
##         try:
##             return self.__keywords__[name]
##         except KeyError:
##             raise AttributeError(name)
# }}}

# {{{ class Symbol(str):
class Symbol(str):

    def __str__(self):
        return str.__str__(self)

    def __pprint__(self, indent, parens=False):
        return str.__str__(self)

    def asStr(self):
        return str.__str__(self)

    def __repr__(self):
        return "~" + str.__str__(self)
# }}}

# {{{ def opfields(obj):
def opfields(obj):
    operands = getattr(obj, "__operands__", None)
    return operands and operands.fields or {}
# }}}

# {{{ def opargs(obj):
def opargs(obj):
    operands = getattr(obj, "__operands__", [])
    return tuple(operands.elems)
# }}}
import language
