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

from util import attrdict

try:
    from livedesk.util import debug, debugmode, dp
except ImportError: pass
# }}}

# {{{ class Symbol(object):
class Symbol(object):

    def fromStr(s):
        i = s.find(":")
        if i == -1:
            return Symbol("", s)
        else:
            return Symbol(s[:i], s[i+1:])
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
location = Symbol("s", "location")

# {{{ class Location(tuple):
class Location(tuple):

    _pathRx = re.compile("(\.|\/)")
    property = object()
    element = object()
    elementId = object()
    content = object()
    tag = object()
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
            for k, s in res[:-1]:
                if k is Location.tag:
                    raise ValueError, "invalid location (tag must be the last part)"
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
    
    # {{{ def __str__(self):
    def __str__(self):
        return self.__pprint__(0, False)
    # }}}
        
    # {{{ def __pprint__(self, indent, withMark=True):
    def __pprint__(self, indent, withMark=True):
        parts = []
        for k, p in self:
            if k == Location.property:
                parts.append(".%s" % p)
            elif k == Location.content:
                parts.append("/*")
            elif k == Location.tag:
                parts.append(".[tag]")
            elif k == Location.elementId:
                parts.append("/&%s" % p)
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

        res = Location(tuple.__add__(self, other))
        res._absolute = self._absolute
        return res
    # }}}

    # {{{ def __eq__(self, other):
    def __eq__(self, other):
        return (isinstance(other, Location)
                and tuple.__eq__(self, other)
                and self._absolute == other._absolute)
    # }}}

    # {{{ def __ne__(self, other):
    def __ne__(self, other):
        return ((not isinstance(other, Location))
                or tuple.__ne__(self, other)
                or self._absolute != other._absolute)
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

# {{{ notify = 
notify = attrdict(valueChanged = Symbol("", "value-changed"),
                  addProperty = Symbol("", "add-property"),
                  delProperty = Symbol("", "del-property"),
                  addSlice = Symbol("", "add-slice"),
                  delSlice = Symbol("", "del-slice"),
                  tagChanged = Symbol("", "tag-changed"),
                  noSuchLocation = Symbol("", "no-such-location"))
# }}}

# {{{ class BaseDoc(object):

# {{{ def strToSymbol(x):
def strToSymbol(x):
    if isinstance(x, str):
        return Symbol("", x)
    else:
        return x
# }}}
    
class BaseDoc(object):

    # {{{ Base Methods (abstract)

    # Everything else is implemented in terms of these
    
    def propertyNames(self):
        return []

    def _hasProperty(self, name):
        return name in self.propertyNames()

    def _getProperty(self, name):
        return None

    def _setProperty(self, name, val):
        raise NotImplementedError

    def _delProperty(self, name):
        raise NotImplementedError

    def contentLen(self):
        return 0

    def getElement(self, index):
        return None

    def setElement(self, index):
        raise NotImplementedError

    # The default behavior for IDs is that we assume the content is read-only
    # and use indexs for IDs
    # TODO: Put this in a ReadOnlyContentBaseDoc subclass

    def elementIds(self):
        return [Symbol("", str(i)) for i in range(self.contentLen())]

    def getElementId(self, index):
        return Symbol("", str(index))

    def getElementById(self, id):
        return self.getElement(id)
    
    def setElementById(self, id, val):
        return setElement(index, val)

    def hasElementId(self, id):
        return id in self.elementIds()

    def delElement(self, index):
        raise NotImplementedError

    def delSlice(self, index, length):
        raise NotImplementedError

    def append(self, val):
        raise NotImplementedError

    def insert(self, index, val):
        raise NotImplementedError

    def splice(self, index, val):
        raise NotImplementedError

    def getTag(self):
        return doc

    def setTag(self):
        raise NotImplementedError

    # }}}

    # {{{ Watcher methods (abstract)
    def addPropertyWatcher(self, property, watcher):
        raise NotImplementedError

    def removePropertyWatcher(self, property, watcher):
        raise NotImplementedError

    def addElementWatcher(self, index, watcher):
        raise NotImplementedError

    def removeElementWatcher(self, index, watcher):
        raise NotImplementedError

    def addElementIdWatcher(self, id, watcher):
        raise NotImplementedError

    def removeElementIdWatcher(self, id, watcher):
        raise NotImplementedError

    def addTagWatcher(self, watcher):
        raise NotImplementedError

    def removeTagWatcher(self, watcher):
        raise NotImplementedError

    def addDocWatcher(self, watcher):
        raise NotImplementedError

    def removeDocWatcher(self, watcher):
        raise NotImplementedError

    # {{{ Internal notification methods

    # Property watchers

    def _notifyPropertyChanged(self, property):
        raise NotImplementedError

    def _notifyPropertyRemoved(self, property):
        raise NotImplementedError

    # Element watchers (by ID)

    def _notifyElementChanged(self, id):
        raise NotImplementedError

    def _notifyElementChangedAtIndex(self, index):
        raise NotImplementedError
    
    def _notifyElementRemoved(self, id):
        raise NotImplementedError

    # Tag watchers

    def _notifyTagChanged(self, tag):
        raise NotImplementedError

    # Doc Watchers

    def _notifyAddProperty(self, property):
        raise NotImplementedError

    def _notifyDelProperty(self, property):
        raise NotImplementedError

    def _notifyAddSlice(self, index, length):
        raise NotImplementedError

    def _notifyDelSlice(self, index, length):
        raise NotImplementedError

    # }}}

    # }}}

    # {{{ strToSymbol wrappers (property methods)
    def hasProperty(self, name):
        return self._hasProperty(strToSymbol(name))

    def getProperty(self, name):
        return self._getProperty(strToSymbol(name))

    def setProperty(self, name, val):
        self._setProperty(strToSymbol(name), val)

    def delProperty(self, name):
        self.delProperty(strToSymbol(name))
    # }}}

    # {{{ def extend(self, other):
    def extend(self, other):
        if isinstance(other, BaseDoc):
            for name, val in other.properties():
                self.setProperty(name, val)

            for val in other.content():
                self.append(val)
            
        elif isinstance(other, dict):
            for name, val in other.items():
                self.setProperty(name, val)
            
        else:
            for val in other:
                self.append(val)
    # }}}

    # {{{ def copy(self):
    def copy(self):
        def cp(x):
            f = getattr(x, "copy", None)
            if f == None:
                return x
            else:
                return f()
        elems = [cp(self.getElement(i)) for i in range(self.contentLen())]
        props = dict([(name, cp(self.getProperty(name))) for name in self.propertyNames()])
        return Doc(self.tag, elems, props)
    # }}}

    # {{{ def getByTag(self, tag):
    def getByTag(self, tag):
        res = []
        for i in range(self.contentLen()):
            e = self.getElement(i)
            if getattr(e, "tag", None) == tag:
                res.append(e)
        return res
    # }}}

    # {{{ def __getitem__(self, key):
    def __getitem__(self, key):
        if isinstance(key, Symbol):
            return self.getProperty(key)
        
        elif isinstance(key, str):
            return self.getProperty(Symbol("", key))

        else:
             # Note some implementations may accept slices 
            return self.getElement(key)
    # }}}
        
    # {{{ def __setitem__(self, key, val):
    def __setitem__(self, key, val):
        if isinstance(key, Symbol):
            self.setProperty(key, docValue(val))
        elif isinstance(key, str):
            self.setProperty(Symbol("", key), docValue(val))
        else:
            self.setElement(key, docValue(val))
    # }}}

    # {{{ def __delitem__(self, key):
    def __delitem__(self, key):
        if isinstance(key, Symbol):
            self.delProperty(key)
        
        elif isinstance(key, str):
            self.delProperty(Symbol("", key))

        else:
            # Note some implementations may accept slices here
            self.delElement(key)
    # }}}

    # {{{ def __len__(self):
    def __len__(self):
        return len(self._d.elementIds)
    # }}}

    # {{{ property: tag
    def _setTag(self, tag):
        if isinstance(tag, Symbol):
            self.setTag(tag)
        elif isinstance(tag, str):
            self.setTag(Symbol("", tag))
        else:
            raise TypeError, "Tag must be a symbol"

    tag = property(lambda self: self.getTag(), _setTag)
    # }}}

    # {{{ def content(self):
    def content(self):
        for x in range(self.contentLen()):
            yield self.getElement(x)
    # }}}

    # {{{ def properties(self):
    def properties(self):
        for name in self.propertyNames():
            yield name, self.getProperty(name)
    # }}}

    # {{{ def propertyValues(self):
    def propertyValues(self):
        for name in self.propertyNames():
            yield self.getProperty(name)
    # }}}

    # {{{ def getLoc(self, loc):
    def getLoc(self, loc):
        res = self
        for k, l in loc:
            if isinstance(res, Location):
                res = self.getLoc(res)
            
            if k == Location.property:
                if not isinstance(res, BaseDoc):
                    raise NoSuchLocationError, loc
                res = res.getProperty(l)
                if res == None:
                    raise NoSuchLocationError, loc
                    
            elif k == Location.element:
                if isinstance(l, int):
                    res = res.getElement(l)
                elif isinstance(l, Symbol):
                    for e in res.content():
                        if isinstance(e, BaseDoc) and e.tag == l:
                            res = e
                            break
                    else:
                        raise ValueError, "invalid location"

            elif k == Location.elementId:
                res = res.getElementById(l)

            elif k == Location.tag:
                res = res.getTag()

            else:
                raise ValueError, "invalid location"
                    
        return res
    # }}}
    
    # {{{ def setLoc(self, loc, val):
    def setLoc(self, loc, val):
        x = self.getLoc(loc[:-1])

        if not isinstance(x, BaseDoc):
            raise NoSuchLocationError, loc

        k, name = loc[-1]
        if k == Location.property and isinstance(name, Symbol):
            x[name] = val
        elif k == Location.element and isinstance(name, int):
            x[name] = val
        elif k == Location.tag:
            x.setTag(val)
        else:
            raise ValueError, "invalid set location", loc
            
    # }}}
    
    # {{{ def get(self, key, default=None):
    def get(self, key, default=None):
        if isinstance(key, str):
            key = Symbol("", key)
        res = self.getProperty(key)
        if res == None:
            return default
        return res
    # }}}

    # {{{ def walk(self, f):
    def walk(self, f):
        for x in self.propertyValues():
            f(x)
            if isinstance(x, BaseDoc):
                x.walk(f)
        for x in self.content():
            f(x)
            if isinstance(x, Doc):
                x.walk(f)
    # }}}

    # {{{ def breakContent(self, index):
    def breakContent(self, index):
        # TODO: We could be a lot more sensible here
        # e.g. what if self[index-1] is not a doc!
        if index == 0:
            val = plaindoc()
        else:
            val = doc(self[index-1].tag)
        self.insert(index, val)
    # }}}

    # {{{ def getElementIndex(self, id):
    def getElementIndex(self, id):
        for i, eid in enumerate(self.elementIds()):
            if id == eid:
                return i
        else:
            return -1
    # }}}

# }}}

# {{{ class Doc(BaseDoc):
class _Doc(object):
    pass

class Doc(BaseDoc):
    
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
        self._d.holder = None
        self._d.tag = tag
        self._d.properties = {}
        self._d.elements = {}
        self._d.elementIds = []
        self._d.nextId = 0

        for arg in args:
            self.extend(arg)
    # }}}

    # {{{ def _nextElementId(self):
    def _nextElementId(self):
        res = Symbol("", str(self._d.nextId))
        self._d.nextId += 1
        return res
    # }}}

    # {{{ def _addProperties(self, d):
    def _addProperties(self, d):
        for name, val in d.items():
            self[name] = docValue(val)
    # }}}
        
    # {{{ def _appendContent(self, seq):
    def _appendContent(self, seq):
        for x in seq:
            id = self._nextElementId()
            self._d.elements[id] = docValue(x)
            self._d.elementIds.append(id)
    # }}}

    # {{{ def _verify(self):
    def _verify(self):
        if not self._d:
            raise UndefinedDocError()
    # }}}

    # {{{ BASE METHODS
            
    # {{{ def propertyNames(self):
    def propertyNames(self):
        self._verify()
        return self._d.properties.iterkeys()
    # }}}

    # {{{ def _hasProperty(self, name):
    def _hasProperty(self, name):
        self._verify()
        return name in self._d.properties
    # }}}    

    # {{{ def _getProperty(self, name):
    def _getProperty(self, name):
        self._verify()
        return self._d.properties.get(name)
    # }}}

    # {{{ def _setProperty(self, name, val):
    def _setProperty(self, name, val):
        self._verify()
        self._d.properties[name] = docValue(val)
    # }}}

    # {{{ def _delProperty(self, name):
    def _delProperty(self, name):
        self._verify()
        del self._d.properties[name]
    # }}}

    # {{{ def contentLen(self):
    def contentLen(self):
        self._verify
        return len(self._d.elementIds)
    # }}}

    # {{{ def getElement(self, index):
    def getElement(self, index):
        self._verify()
        if isinstance(index, slice):
            ids = self._d.elementIds[index]
            return [self._d.elements[id] for id in self._d.elementIds[index]]
        else:
            return self._d.elements[self._d.elementIds[index]]
    # }}}

    # {{{ def getElementByID(self, id):
    def getElementByID(self, id):
        self._verify()
        return self._d.elements[id]
    # }}}

    # {{{ def getElementId(self, index):
    def getElementId(self, index):
        self._verify()
        return self._d.elementIds[index]
    # }}}

    # {{{ def setElement(self, index, val):
    def setElement(self, index, val):
        self._verify()
        self._d.elements[self._d.elementIds[index]] = docValue(val)
    # }}}

    # {{{ def delElement(self, index):
    def delElement(self, index):
        self._verify()
        if isinstance(index, slice):
            for id in self._d.elementIds[index]:
                del self._d.elements[id]
        else:
            del self._d.elements[self._d.elementIds[index]]
        del self._d.elementIds[index]
    # }}}

    # {{{ def elementIds(self):
    def elementIds(self):
        self._verify()
        return iter(self._d.elementIds)
    # }}}

    # {{{ def getTag(self):
    def getTag(self):
        self._verify()
        return self._d.tag
    # }}}

    # {{{ def setTag(self, tag):
    def setTag(self, tag):
        self._verify()
        self._d.tag = tag
    # }}}

    # {{{ def append(self, x):
    def append(self, x):
        self._verify()
        id = self._nextElementId()
        self._d.elements[id] = docValue(x)
        self._d.elementIds.append(id)
    # }}}
    
    # {{{ def insert(self, index, val):
    def insert(self, index, val):
        self._verify()
        id = self._nextElementId()
        self._d.elements[id] = docValue(val)
        self._d.elementIds.insert(index, id)
    # }}}
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
        for x in self.content():
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
    
    # {{{ def isHeld(self):
    def isHeld(self):
        self._verify()
        return self._d.holder is not None
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
        if other is self:
            return True
        
        if isinstance(other, Doc):
            other._verify()
            return (self._d.tag == other._d.tag
                and self._d.properties == other._d.properties
                and list(self.content()) == list(other.content()))
        else:
            return False
    # }}}

    # {{{ def __ne__(self, other):
    def __ne__(self, other):
        self._verify()
        if other is self:
            return False
        
        if isinstance(other, Doc):
            other._verify()
            return (self._d.properties != other._d.properties
                    or self._d.content != other._d.content)
        else:
            return True
    # }}}

    # {{{ def __nonzero__(self):
    def __nonzero__(self):
        return True
    # }}}

    # {{{ def applyTo(self, f):
    def applyTo(self, f, *args):
        self._verify()
        kws = {}
        for name, val in self._d.properties.items():
            if name.namespace == "":
                kws[name.name] = val
        return f(*(list(args) + list(self.content())), **kws)
    # }}}

    # {{{ def combine self other:
    def combine(self, other):
        # TODO: Here is where we implement something akin to classes and objects,
        # or, a better fit, prototype references in prototype based OO languages.
        # The doc "combined" with self becomes a part of it in the sense that
        # properties not found in self are looked up in the combined doc.
        # Elements are probably not combined.
        # A single doc may be combined into many docs, so that, e.g. a collection
        # of channels providing operations on some kind of dock can be shared by all
        # those docs. Unlike OO languages, this sharing cannot be a means to
        # invisible dependencies. Therefore, `other` must be immutable. If it is not,
        # an immutable version is automatically combined instead.
        #
        # This is all TODO. For now we just copy `other` in.

        for n in other.propertyNames():
            if not self.hasProperty(n):
                self.setProperty(n, other.getProperty(n))
        return self # This allows us to do stuff like `x = D[...].combine foo`
    # }}}


# }}}

# {{{ class StdWatchable(object):
class StdWatchable(object):

    # Q: Should we have separate mixins for docs that support
    # property/element watchers and docs that support doc-watchers?

    def __init__(self):
        self._propertyWatchers = {} # property-name -> [watchers]
        self._elementWatchers = {}  # id -> [watchers]
        self._tagWatchers = []
        self._docWatchers = []

    def addPropertyWatcher(self, property, watcher):
        watchers = self._propertyWatchers.setdefault(property, [])
        watchers.append(watcher)

    def removePropertyWatcher(self, property, watcher):
        self._propertyWatchers.get(property).remove(watcher)

    def addElementIdWatcher(self, id, watcher):
        watchers = self._elementWatchers.setdefault(id, [])
        watchers.append(watcher)

    def removeElementIdWatcher(self, id, watcher):
        self._elementWatchers.get(id).remove(watcher)

    def addTagWatcher(self, watcher):
        self._tagWatchers.append(watcher)

    def removeTagWatcher(self, watcher):
        self._tagWatchers.remove(watcher)
        
    def addDocWatcher(self, watcher):
        self._docWatchers.append(watcher)

    def removeDocWatcher(self, watcher):
        self._docWatchers.remove(watcher)

    # {{{ Internal notification methods

    # {{{ property watchers
    def _notifyPropertyChanged(self, property):
        watchers = self._propertyWatchers.get(property)
        if watchers:
            for w in watchers:
                w[notify.valueChanged]()

    def _notifyPropertyRemoved(self, property):
        watchers = self._propertyWatchers.get(property)
        if watchers:
            for w in watchers:
                w[notify.noSuchLocation]()
    # }}}

    # {{{ element watchers
    def _notifyElementChanged(self, id):
        watchers = self._elementWatchers.get(id)
        if watchers:
            for w in watchers:
                w[notify.valueChanged]()
                
    def _notifyElementChangedByIndex(self, index):
        self._notifyElementChanged(self.getElementId(index))

    def _notifyElementRemoved(self, id):
        watchers = self._elementWatchers.get(id)
        if watchers:
            for w in watchers:
                w[notify.noSuchLocation]()
    # }}}

    # {{{ doc watchers
    def _notifyAddProperty(self, property):
        for w in self._docWatchers:
            w[notify.addProperty](property)

    def _notifyDelProperty(self, property):
        for w in self._docWatchers:
            w[notify.delProperty](property)
        
    def _notifyAddSlice(self, index, length):
        for w in self._docWatchers:
            w[notify.addSlice](index, length)

    def _notifyDelSlice(self, index, length):
        for w in self._docWatchers:
            w[notify.delSlice](index, length)

    def _notifyTagChanged(self, tag):
        for w in self._docWatchers:
            w[notify.tagChanged](tag)
        for w in self._tagWatchers:
            w[notify.valueChanged]()
    # }}}

    # }}}

# }}}

class WatchableDoc(StdWatchable, BaseDoc):
    pass

# {{{ class WDoc(Doc, StdWatchable):
class WDoc(StdWatchable, Doc):

    def __init__ (self, doc):
        Doc.__init__(self, doc.tag)
        StdWatchable.__init__(self)
        def change(x):
            if isinstance(x, Doc):
                return WDoc(x)
            else:
                return x
            
        for x in doc.content():
            self.append(change(x))
        for name, x in doc.properties():
            self[name] = change(x)

    # {{{ def _setProperty(self, name, val):
    def _setProperty(self, name, val):
        old = self._getProperty(name)
        Doc._setProperty(self, name, val)
        if val is not old:
            self._notifyPropertyChanged(name)
    # }}}

    # {{{ def _delProperty(self, name):
    def _delProperty(self, name):
        Doc._delProperty(self, name)
        self._notifyPropertyRemoved(name)
    # }}}

    # {{{ def setElement(self, index, val):
    def setElement(self, index, val):
        old = self.getElement(index)
        Doc.setElement(self, index, val)
        if val is not old:
            self._notifyElementChanged(self.getElementId(index))
    # }}}

    # {{{ def delElement(self, index):
    def delElement(self, index):
        # `index` could be a slice

        if isinstance(index, slice):
            start = index.start
            if start == None:
                start = 0
            stop = index.stop
            if stop == None:
                stop = self.contentLen()

            removedIds = [self.getElementId(i) 
                          for i in range(start, stop)]
                
            Doc.delElement(self, index)

            for id in removedIds:
                self._notifyElementRemoved(id)

            self._notifyDelSlice(start, stop - start)
        else:
            id = self.getElementId(index)
            Doc.delElement(self, index)
            self._notifyElementRemoved(id)
            self._notifyDelSlice(index, 1)
    # }}}

    # {{{ def setTag(self, tag):
    def setTag(self, tag):
        old = self.getTag()
        Doc.setTag(self, tag)
        if old is not tag:
            self._notifyTagChanged(tag)
    # }}}

    # {{{ def append(self, x):
    def append(self, x):
        Doc.append(self, x)
        self._notifyAddSlice(self.contentLen(), 1)
    # }}}
    
    # {{{ def insert(self, index, val):
    def insert(self, index, val):
        Doc.insert(self, index, val)
        self._notifyAddSlice(index, 1)
    # }}}

    # {{{ def extend(self, other):
    def extend(self, other):
        oldProps = self.propertyNames()
        oldLen = self.contentLen()
        Doc.extend(self, other)
        addedProps = set(self.propertyNames()) - set(oldProps)
        for p in addedProps:
            self._notifyAddProperty(p)
        self._notifyAddSlice(oldLen, self.contentLen() - oldLen)
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
        if x.isHeld():
            raise ValueError, "doc is already located:\n" % x
        else:
            return x

    elif hasattr(x, "__iter__"):
        d = plaindoc()
        d._appendContent(x)
        return d

    elif isinstance(x, types.FunctionType):
        # TODO: This is a temporary fix for allowing channel ends in docs
        return x

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
    if isinstance(x, (BaseDoc, Symbol)):
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
    if isinstance(x, BaseDoc):
        return x.tag
    else:
        return typeSymbols[type(x)]
# }}}

import language
