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
import operator

# {{{ def record(**keywords):
def record(**keywords):
    class Record:
        def __repr__(self):
            return "Record(%s)" % ' '.join(["%s=%s" % (k,v)
                                            for k,v in self.__dict__.items()])
        
    r = Record()
    r.__dict__.update(keywords)
    return r
# }}}

# {{{ def concat(lists):
def concat(lists):
    return reduce(operator.add, lists, [])
# }}}

# {{{ debug stuff
debugMode=0
def debugmode(x):
    global debugMode
    debugMode = x
    # print 'DEBUGMODE', x


def debug(mode=None):
    if mode == None or debugMode == mode:
        import mydb
        mydb.set_trace()

def dp(s, mode):
    if debugMode == mode:
        print s
# }}}

# {{{ class attrdict(dict):
class attrdict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError, name

    def __setattr__(self, name, value):
        self[name] = value
# }}}
