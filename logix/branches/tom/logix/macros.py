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

import rootops, language, parser

from data import *

try:
    from livedesk.util import debug, debugmode, dp
    import mydb
except ImportError: pass
# }}}

gensymcounter=0

# {{{ class splice(object):
class splice(object):

    def __init__(self, items):
        self.items = items

    def __repr__(self):
        return "<splice: %s>" % self.items
# }}}

# {{{ def expand(obj):
def expand(obj):
    #try:
    #    return _expand(obj, {})
    #except SyntaxError:
    #    raise SyntaxError("invalid quote escape %s" % obj)
    return _expand(obj, {})
# }}}

# {{{ def _expand(obj, context, quoteDepth=0):
def _expand(obj, context, quoteDepth=0, lineno=None):
    ln = getattr(obj, "lineno", None)
    if ln != None:
        lineno = ln
    
    if isinstance(obj, Doc):
        op = language.getOp(obj.tag)
        if quoteDepth == 0 and op != None and hasattr(op, 'macro'):
            moduleContext = context.setdefault(op.language.__module__, MacroContext())
            moduleContext.beginScope()
            expanded, _ = expand1(obj, op, moduleContext)
            assert expanded is not obj
            res = _expand(expanded, context, lineno=lineno)
            moduleContext.endScope()

        else:
            if obj.tag == rootops.quote:
                docQuoteDepth = quoteDepth + 1
                
            elif obj.tag == rootops.escape:
                extra = obj.get("extra")
                if extra != None:
                    escapeLevel = extra.contentLen() + 1
                else:
                    escapeLevel = 1
                if escapeLevel > quoteDepth:
                    # TODO: Proper filename in error message
                    raise SyntaxError, ("quote escape greater than quote nesting", (None, lineno,0, ""))
                elif escapeLevel == quoteDepth:
                    docQuoteDepth = 0
                else:
                    docQuoteDepth = quoteDepth

            else:
                docQuoteDepth = quoteDepth

            content = []
            for x in obj.content():
                ex = _expand(x, context, docQuoteDepth, lineno=lineno)
                if isinstance(ex, splice):
                    content.extend(ex.items)
                else:
                    content.append(ex)
            
            properties = {}
            for name, val in obj.properties():
                ex = _expand(val, context, docQuoteDepth, lineno=lineno)
                if isinstance(ex, splice):
                    raise SyntaxError, "cannot splice into named operand '%s'" % name
                properties[name] = ex

            res = Doc(obj.tag, content, properties)
            parser.copySourcePos(obj, res)

    else:
        res = obj

    return res
# }}}

# {{{ def expand1(doc, op, context):
def expand1(doc, op, context):
    if hasattr(op, 'macro'):
        lineno = getattr(doc, 'lineno', None)
        try:
            expansion = op.expandMacro(doc, context)
        except:
            exc_info = sys.exc_info()
            err = SyntaxError("macro expansion failed for:\n%s\n%s" % (doc, exc_info[1]),
                              "file??", lineno or -1, None, None)
            err.nested = exc_info
            raise err

        if lineno:
            try:
                expansion.lineno=lineno
            except: pass

        return expansion, True

    else:
        return op, False
# }}}

# {{{ def gensym():
def gensym(name=''):
    global gensymcounter
    gensymcounter += 1
    return Symbol("", "#%s%s" % (name, gensymcounter))
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

    def __delitem__(self, key):
        for scope in self.scopes[::-1]:
            if key in scope:
                del scope[key]
                break

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

