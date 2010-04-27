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
import sys
import inspect

import rootops
from language import \
     Operator, opargs, opfields, OperatorType

from data import *

try:
    from livedesk.util import debug, debugmode, dp
    import mydb
except ImportError: pass
# }}}

gensymcounter=0

# {{{ def expand(obj):
def expand(obj):
    try:
        return _expand(obj, {})
    except SyntaxError:
        raise SyntaxError("invalid quote escape %s" % obj)
# }}}

# {{{ def _expand(obj, context, quoteDepth=0):
def _expand(obj, context, quoteDepth=0):
    typ = type(obj)

    opclass = obj.__class__
    if quoteDepth == 0 and hasattr(opclass, '__macro__'):
        moduleContext = context.setdefault(opclass.__module__, MacroContext())
        moduleContext.beginScope()
        expanded, _ = expand1(obj, moduleContext)
        assert expanded is not obj
        res = _expand(expanded, context)
        moduleContext.endScope()

    elif isinstance(typ, OperatorType):
        if isinstance(obj, rootops.quote):
            operands = _expand(obj.__operands__, context, quoteDepth+1)
        elif (isinstance(obj, rootops.escape) and
              (1 + len(obj.__operands__.get('extra') or [])) >= quoteDepth):
            operands = _expand(obj.__operands__, context, 0)
        else:
            operands = _expand(obj.__operands__, context, quoteDepth)

        res = typ(*operands.elems, **operands.fields)
        copymeta(obj, res)

    elif isinstance(obj, rootops.quote):
        res = rootops.quote(_expand(obj[0], context, 1))
        copymeta(obj, res)

    elif isinstance(obj, flist):
        xlist = [_expand(x, context, quoteDepth) for x in obj.elems]
        xfields = dict([(k,_expand(v, context, quoteDepth)) for k,v in obj.items()])
        res = flist.new(xlist, xfields)
        copymeta(obj, res)

    elif typ == list:
        res = [_expand(x, context, quoteDepth) for x in obj]

    elif typ == tuple:
        res = tuple([_expand(x, context, quoteDepth) for x in obj])
        
    elif typ == dict:
        res = dict([(_expand(k, context, quoteDepth),
                     _expand(v, context, quoteDepth))
                    for k,v in obj.items()])

    else:
        res = obj

    return res
# }}}

# {{{ def expand1(op, context):
def expand1(op, context):
    if hasattr(op.__class__, '__macro__'):
        try:
            expansion = op.__macro__(context)
        except:
            exc_info = sys.exc_info()
            raise MacroExpandError(op, exc_info)

        lineno = getmeta(op, 'lineno')
        if lineno:
            try:
                setmeta(expansion, lineno=lineno)
            except: pass

        return expansion, True

    else:
        return op, False
# }}}

# {{{ def gensym():
def gensym(name=''):
    global gensymcounter
    gensymcounter += 1
    return Symbol("#%s%s" % (name, gensymcounter))
# }}}

# {{{ class MacroExpandError(Exception):
class MacroExpandError(Exception):

    def __init__(self, op, exc_info):
        self.op = op
        self.exc_info = exc_info
        self.args = exc_info[1].args

    def __str__(self):
        lineno = getmeta(self.op, 'lineno')
        line = lineno and ("\nline: %s" % lineno) or ""
        opstr = str(self.op)
        if len(opstr)>200:
            opstr = opstr[:200] + "..."
        return "Macro expand error: %s: %s\nMacro:\n%s%s" % (
            self.exc_info[0].__name__,
            self.exc_info[1],
            opstr,
            line)

    def pm(self):
        import mydb
        mydb.pm(self.exc_info[2])
# }}}

# {{{ def formatMacroError(error, filename=None):
def formatMacroError(error, filename=None):
    lineno = getmeta(error.op, 'lineno')
    if lineno and filename:
        f = file(filename)
        for i in xrange(lineno):
            line = f.readline()
        line = "--> %d %s\n" % (lineno, line.strip())
    else:
        line = ''
    opstr = str(error.op)
    if len(opstr)>200:
        opstr = opstr[:200] + "..."
    print "Macro expand error: %s\n%s%s" % (error.exc_info[1], line, opstr)
# }}}

# {{{ class MacroContext:
class MacroContext:

    def __init__(self):
        self.scopes = [{}]
        
    def setGlobal(self, name, val):
        self.scopes[0][name] = val

    def __setitem__(self, name, val):
        self.scopes[-1][name] = val

    def __getitem__(self, key):
        for scope in self.scopes[::-1]:
            res = scope.get(key)
            if res is not None:
                return res
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def beginScope(self):
        self.scopes.append({})
        
    def endScope(self):
        del self.scopes[-1]
# }}}
