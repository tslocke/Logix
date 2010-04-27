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
import re
import types

try:
    from livedesk.util import debug, debugmode, dp
except ImportError: pass
# }}}

# {{{ class Symbol(object):
class Symbol(object):

    def fromStr(s):
        bits = s.strip().split(":")
        if len(bits) == 1:
            return Symbol("", bits[0])
        else:
            return Symbol(bits[0], bits[1])
    fromStr = staticmethod(fromStr)

    def __init__(self, ns, sym):
        assert isinstance(ns, str)
        assert isinstance(sym, str)
        self._ns = intern(ns)
        self._sym = intern(sym)

    def __repr__(self):
        if self._ns == "":
            return self._sym
        else:
            return "%s:%s" % (self._ns, self._sym)

    def __nonzero__(self):
        return self != false and self != none

    def __eq__(self, other):
        return (isinstance(other, Symbol)
                and other._ns == self._ns
                and other._sym == self._sym)

    def __ne__(self, other):
        return ((not isinstance(other, Symbol))
                or other._ns != self._ns
                or other._sym != self._sym)

    def copy(self):
        return Symbol(self._ns, self._sym)

    def __hash__(self):
        return hash(repr(self))

    name = property(lambda self: self._sym)
    namespace = property(lambda self: self._ns)

# }}}

true = Symbol("s", "true")
false = Symbol("s", "false")
none = Symbol("s", "none")
doc = Symbol("s", "doc")

# {{{ class Location(tuple):
class Location(tuple):

    _pathRx = re.compile("(\.|\/)")
    property = object()
    element = object()
    content = object()
    all = object()

    # {{{ def __new__ cls locstr:
    def __new__(cls, locstr=None):
        if locstr == None:
            res = tuple.__new__(cls)
            res._absolute = False
        elif isinstance(locstr, (Location)):
            res = tuple.__new__(cls, locstr)
            res._absolute = locstr._absolute
        elif isinstance(locstr, (tuple, list)):
            res = tuple.__new__(cls, locstr)
            res._absolute = False
        else:
            err = ValueError("invalid location: %s" % locstr)

            if locstr.startswith("//") or locstr.startswith("/."):
                abs = True
                locstr = locstr[1:]
            else:
                abs = False

            if locstr[0] not in "./":
                raise err

            parts = Location._pathRx.split(locstr)
            del parts[0]

            if len(parts) == 0:
                raise err

            i = 0
            location = []
            while i < len(parts):
                if len(parts) < i + 2:
                    raise err

                loc = parts[i+1]

                if parts[i] == ".":
                    if loc == "*":
                        location.append( (Location.property, Location.all) )
                    else:
                        location.append( (Location.property, Symbol.fromStr(loc)) )

                elif parts[i] == "/":                
                    if loc == "*":
                        location.append( (Location.content, Location.all) )
                    else:
                        try:
                            location.append( (Location.element, int(loc)) )
                        except ValueError:
                            location.append( (Location.element, Symbol.fromStr(loc)) )
                else:
                    raise err
                i += 2
            res = tuple.__new__(cls, location)
            res._absolute = abs
        return res
    # }}}

    # {{{ def __repr__(self):
    def __repr__(self):
        return self.__pprint__(0, True)
    # }}}
        
    # {{{ def __pprint__(self, indent, withMark=True):
    def __pprint__(self, indent, withMark=True):
        parts = []
        for k, p in self:
            if k == Location.property:
                parts.append(".%s" % p)
            elif k == Location.content:
                parts.append("/*")
            else:
                parts.append("/%s" % p)

        abs = self._absolute and "/" or ""
            
        if withMark:
            return "@" + abs + "".join(parts)
        else:
            return abs + "".join(parts)
    # }}}
    
    # {{{ def __getitem__(self, key):
    def __getitem__(self, key):
        if isinstance(key, slice):
            return Location(tuple.__getitem__(self, key))
        else:
            return tuple.__getitem__(self, key)
    # }}}

    # {{{ def __add__(self, other):
    def __add__(self, other):
        if not isinstance(other, Location):
            raise TypeError, "can only add location to location (not %s)" % type(other)

        return Location(tuple.__add__(self, other))
    # }}}

    def isAbsolute(self):
        return self._absolute

    def asAbsolute(self):
        l = Location(self)
        l._absolute = True
        return l
    
# }}}

typeSymbols = {int:      Symbol("s", "integer"),
               long:     Symbol("s", "integer"),
               float:    Symbol("s", "number"),
               str:      Symbol("s", "text"),
               Symbol:   Symbol("s", "symbol"),
               Location: Symbol("s", "location")}

class NoSuchLocationError(Exception):
    pass

class UndefinedDocError(Exception):
    pass

# {{{ class Doc(object):
class _Doc(object):
    pass

class Doc(object):
    
    # {{{ def new(tag, *args, **kws):
    def new(tag, *args, **kws):
        d = Doc(tag)
        d._appendContent(args)
        d._addProperties(kws)
        return d
    new = staticmethod(new)
    # }}}

    # {{{ def __init__(self, tag, *args)
    def __init__(self, tag, *args):
        if isinstance(tag, str):
            tag = Symbol.fromStr(tag)
        else:
            assert isinstance(tag, Symbol)
            if tag in typeSymbols.values():
                raise ValueError, "%s is a reserved symbol" % tag
        
        self._d = _Doc()
        self._d.location = None
        self._d.tag = tag
        self._d.properties = {}
        self._d.content = []

        for arg in args:
            self.extend(arg)
    # }}}

    # {{{ def _addProperties(self, d):
    def _addProperties(self, d):
        for name, val in d.items():
            self[name] = docValue(val)
    # }}}
        
    # {{{ def _appendContent(self, seq):
    def _appendContent(self, seq):
        for x in seq:
            self._d.content.append(docValue(x))
    # }}}

    # {{{ def _verify(self):
    def _verify(self):
        if not self._d:
            raise UndefinedDocError()
    # }}}
            
    # {{{ def isLocated(self):
    def isLocated(self):
        self._verify()
        return self._d.location is not None
    # }}}
    
    # {{{ def __getitem__(self, key):
    def __getitem__(self, key):
        self._verify()
        if isinstance(key, Symbol):
            return self._d.properties[key]
        
        elif isinstance(key, str):
            return self._d.properties[Symbol("", key)]

        else:
            # Note this case handles both integers and slices
            return self._d.content[key]
    # }}}
        
    # {{{ def __setitem__(self, key, val):
    def __setitem__(self, key, val):
        self._verify()
        if isinstance(key, Symbol):
            self._d.properties[key] = docValue(val)
        elif isinstance(key, str):
            self._d.properties[Symbol("", key)] = docValue(val)
        else:
            self._d.content[key] = docValue(val)
    # }}}

    # {{{ def __delitem__(self, key):
    def __delitem__(self, key):
        self._verify()
        if isinstance(key, Symbol):
            del self._d.properties[key]
        
        elif isinstance(key, str):
            del self._d.properties[Symbol("", key)]

        else:
            # Note this case handles both integers and slices
            del self._d.content[key]
    # }}}

    # {{{ def getByTag(self, tag):
    def getByTag(self, tag):
        self._verify()
        return Doc(seq, self._getByTag(tag))
        
    def _getByTag(self, tag):
        return [x for x in self._d.content if isinstance(x, Doc) and x.tag == tag]
    # }}}

    # {{{ def getLoc(self, loc):
    def getLoc(self, loc):
        res = self
        for k, l in loc:
            
            if k == Location.property:
                if not isinstance(res, Doc):
                    raise NoSuchLocationError, (self, loc)
                res._verify()
                res = res._d.properties.get(l)
                if res == None:
                    raise NoSuchLocationError, (self, loc)
                    
            elif k == Location.element:
                if not isinstance(res, Doc):
                    raise NoSuchLocationError, (self, loc)
                assert isinstance(l, int)
                res._verify()
                res = res._d.content[l]
                    
            else:
                assert 0
                # {{{ old stuff
                # assert k == Location.content
                #if not res isa Doc:
                #    raise NoSuchLocationError, (self, loc)
                #res._verify()
                #res = res._d.content
                # }}}

        return res
    # }}}
    
    # {{{ def setLoc(self, loc, val):
    def setLoc(self, loc, val):
        x = self
        for k, l in loc[:-1]:
            
            if k == Location.property:
                if not isinstance(x, Doc):
                    raise NoSuchLocationError, (self, loc)
                x._verify()
                x = x._d.properties.get(l)
                if x == None:
                    raise NoSuchLocationError, (self, loc)
                    
            elif k == Location.element:
                if not isinstance(x, Doc):
                    raise NoSuchLocationError, (self, loc)
                assert isinstance(l, int)
                x._verify()
                x = x._d.content[l]
                    
            else:
                raise ValueError, "invalid set location"

        if not isinstance(x, Doc):
            raise NoSuchLocationError, loc

        k, name = loc[-1]
        if k == Location.property:
            x[name] = val
        elif k == Location.element and isinstance(name, int):
            x[name] = val
        else:
            raise ValueError, "invalid set location", loc
            
    # }}}

    # {{{ property: tag
    def _getTag(self):
        self._verify()
        return self._d.tag

    def _setTag(self, tag):
        self._verify()
        if isinstance(tag, Symbol):
            self._d.tag = tag
        elif isinstance(tag, str):
            self._d.tag = Symbol("", tag)
        else:
            raise TypeError, "Tag must be a symbol"

    tag = property(_getTag, _setTag)
    # }}}

    # {{{ def content(self):
    def content(self):
        self._verify()
        return self._d.content.__iter__()
    # }}}

    # {{{ def properties(self):
    def properties(self):
        self._verify()
        return self._d.properties.iteritems()
    # }}}

    # {{{ def propertyNames(self):
    def propertyNames(self):
        self._verify()
        return self._d.properties.iterkeys()
    # }}}

    # {{{ def propertyValues(self):
    def propertyValues(self):
        self._verify()
        return self._d.properties.itervalues()
    # }}}

    # {{{ def __repr__(self):
    def __repr__(self):
        return self.__pprint__(0)
    # }}}

    # {{{ def __pprint__(self, indent, withMark=True):
    def __pprint__(self, indent, withMark=True):
        self._verify()
        
        def pprn(ob, indent):
            if isinstance(ob, (Doc, Location)):
                return ob.__pprint__(indent, False)
            else:
                return repr(ob)

        if self.tag == doc:
            tag = ""
        else:
            tag = repr(self._d.tag)

        newindent = indent + len(tag) + 2
        if withMark:
            newindent += 1
    
        contentstrs = []
        for x in self._d.content:
            contentstrs.append(pprn(x, newindent))
            
        propstrs = []
        for name, val in self._d.properties.items():
            valstr = pprn(val, newindent + len(str(name)) + 1)
            propstrs.append("%s=%s" % (name, valstr))
    
        linelen = 0
        all = contentstrs + propstrs
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
            s = '\n'.join(all[:1] + [indentstr + x for x in all[1:]])
        else:
            s = ' '.join(all)

        if self.tag == doc:
            res = "[%s]" % s
        elif len(s) == 0:
            res = "{%s}" % tag
        else:
            res = "{%s %s}" % (tag, s)
            
        if withMark:
            return "D" + res
        else:
            return res
    # }}}

    # {{{ def get(self, key, default=None):
    def get(self, key, default=None):
        self._verify()
        if isinstance(key, str):
            key = Symbol("", key)
        return self._d.properties.get(key, default)
    # }}}

    # {{{ def hasProperty(self, name):
    def hasProperty(self, name):
        self._verify()
        if isinstance(name, str):
            name = Symbol("", name)
        return name in self._d.properties
    # }}}

    # {{{ def extend(self, other):
    def extend(self, other):
        self._verify()
        if isinstance(other, Doc):
            other._verify()
            self._addProperties(other._d.properties)
            self._appendContent(other._d.content)
            
        elif isinstance(other, dict):
            self._addProperties(other)
            
        else:
            self._appendContent(other)
    # }}}

    # {{{ def copy(self):
    def copy(self):
        self._verify()
        new = Doc(self._d.tag)

        newcontent = new._d.content
        for x in self._d.content:
            newcontent.append(copy(x))

        newprops = new._d.properties
        for name, val in self._d.properties.items():
            newprops[name] = copy(val)

        for name, val in vars(self).items():
            if name == "_d":
                continue
            setattr(new, name, val)

        return new
    # }}}

    # {{{ def append(self, x):
    def append(self, x):
        self._verify()
        self._d.content.append(docValue(x))
    # }}}

    # {{{ def __add__(self, other):
    def __add__(self, other):
        new = self.copy()
        new.extend(other)
        return new
    # }}}

    # {{{ def __radd__(self, other):
    def __radd__(self, other):
        self._verify()
        new = Doc(self._d.tag, list(other))
        new._addProperties(self._d.properties)
        new._appendContent(self._d.content)
        return new
    # }}}

    # {{{ def __eq__(self, other):
    def __eq__(self, other):
        self._verify()
        if isinstance(other, Doc):
            other._verify()
            return (self._d.tag == other._d.tag
                and self._d.properties == other._d.properties
                and self._d.content == other._d.content)
        else:
            return False
    # }}}

    # {{{ def __ne__(self, other):
    def __ne__(self, other):
        self._verify()
        if isinstance(other, Doc):
            other._verify()
            return (self._d.properties != other._d.properties
                    or self._d.content != other._d.content)
        else:
            return True
    # }}}

    # {{{ def contentLen(self):
    def contentLen(self):
        self._verify()
        return len(self._d.content)
    # }}}

    # {{{ def numProperties(self):
    def numProperties(self):
        self._verify()
        return len(self._d.properties)
    # }}}

    # {{{ def __len__(self):
    def __len__(self):
        self._verify()
        return len(self._d.content)
    # }}}

    # {{{ def applyTo(self, f):
    def applyTo(self, f, *args):
        self._verify()
        kws = {}
        for name, val in self._d.properties.items():
            if name.namespace == "":
                kws[name.name] = val
        return f(*(list(args) + self._d.content), **kws)
    # }}}

    # {{{ def __nonzero__(self):
    def __nonzero__(self):
        return True
    # }}}

    # {{{ def walk(self, f):
    def walk(self, f):
        self._verify()
        for x in self._d.properties.values():
            f(x)
            if isinstance(x, Doc):
                x.walk(f)
        for x in self._d.content:
            f(x)
            if isinstance(x, Doc):
                x.walk(f)
    # }}}

# }}}

# {{{ def docValue(x):
def docValue(x):
    if x is False:
        return false

    elif x is True:
        return true

    elif x is None:
        return none

    elif isinstance(x, (str, int, long, float, Symbol, Location)):
        return x

    elif isinstance(x, (list, tuple)):
        d = plaindoc()
        d._appendContent(x)
        return d

    elif isinstance(x, dict):
        d = plaindoc()
        d._addProperties(x)
        return d

    elif isinstance(x, Doc):
        if x.isLocated():
            raise ValueError, "doc is already located:\n" % x
        else:
            return x

    elif hasattr(x, "__iter__"):
        d = plaindoc()
        d._appendContent(x)
        return d

    else:
        raise ValueError, "cannot create doc-value from %s" % x
# }}}

# {{{ def checkDocValue(x):
def checkDocValue(x):
    return isinstance(x, (int, float, long, str, Symbol, Doc))
# }}}

# {{{ def plaindoc(*args):
def plaindoc(*args):
    return Doc(doc, *args)
# }}}

# {{{ def copy(x):
def copy(x):
    if isinstance(x, (Doc, Symbol)):
        return x.copy()
    elif hasattr(x, '__dict__'):
        # subtype of primitive type
        res = type(x)(x)
        res.__dict__.update(x.__dict__)
        return res
    else:
        #assume it's immutable
        return x
# }}}

# {{{ def isDoc(x, symbol):
def isDoc(x, symbol):
    if isinstance(x, Doc):
        if isinstance(symbol, tuple):
            return x.tag in symbol
        else:
            return x.tag == symbol
    else:
        return False
# }}}

# {{{ def typeOf(x):
def typeOf(x):
    # assert x isa docval
    if isinstance(x, Doc):
        return x.tag
    else:
        return typeSymbols[type(x)]
# }}}

import language
