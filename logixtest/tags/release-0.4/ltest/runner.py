# {{{ imports
import sys
import os
import new
import types
import inspect
import traceback
import linecache
import operator

try:
    import IPython.Debugger as ipyDebugger
except ImportError:
    ipyDebugger = None


import logix
from logix.util import attrdict

testlang = logix.imp("ltest.testlang")

try:
    from livedesk.util import dynreload_all, debug
except ImportError:
    dynreload_all = None
    debug = None

try:
    from mydb import pm as postmortem
except ImportError:
    from pdb import pm as postmortem
# }}}
MATCH, NOMATCH, FAIL = range(3)

class ExpectationFailure(Exception): pass

class QuitTests(Exception): pass
class SkipRestOfTest(Exception): pass

# {{{ Expected values
def allTrue(list):
    for e in list:
        if not e:
            return False
    return True

def valueIsExpected(value, expectation):
    t = type(expectation)
    
    if t == types.FunctionType:
        return expectation(value)
    elif t == testlang.Pattern:
        # Hack - right now there's no way to propogate the pattern error message
        # so we just print it
        res = t.test(value)
        if t == True:
            return True
        else:
            print t
            return False
    else:
        return value == expectation

def valuesAreExpected(values, expectations):
    return (len(values) == len(expectations) and
            allTrue(map(valueIsExpected, values, expectations)))

def keywordsAreExpected(keywords, expectations):
    for key, exp in expectations.items():
        if key in keywords and valueIsExpected(keywords[key], exp):
            pass
        else:
            return False
    return True
# }}}

# {{{ Expectations
# {{{ class Expectation:
class Expectation(object):
    pass
# }}}

# {{{ class SimpleExpectation(Expectation):
class SimpleExpectation(Expectation):
    def complete(self, state):
        return state.get(self, False)

    def reset(self, state):
        if state.has_key(self):
            del state[self]

    def needsMore(self, state):
        return not self.complete(state)
# }}}
    
# {{{ class AttrGet(SimpleExpectation):
class AttrGet(SimpleExpectation):

    def __init__(self, name, returnVal):
        self.name = name
        self.returnVal = returnVal

    def test(self, event, state):
        if (isinstance(event, GetAttr) and
            event.name == self.name
            ):
            state[self] = True
            return MATCH, self.returnVal
        else:
            return FAIL, self

    def expecting(self, state):
        return "(get: %s)" % self.name
# }}}

# {{{ class ItemGet(SimpleExpectation):
class ItemGet(SimpleExpectation):

    def __init__(self, key, returnVal):
        self.key = key
        self.returnVal = returnVal

    def test(self, event, state):
        if (isinstance(event, GetItem) and
            event.key == self.key
            ):
            state[self] = True
            return MATCH, self.returnVal
        else:
            return FAIL, self

    def expecting(self, state):
        return "(get-item: %s)" % self.key
# }}}

# {{{ class AttrSet(SimpleExpectation):
class AttrSet(SimpleExpectation):

    def __init__(self, name, expectedVal):
        self.name = name
        self.expectedVal = expectedVal

    def test(self, event, state):
        if (isinstance(event, SetAttr) and
            event.name == self.name and
            valueIsExpected(event.value, self.expectedVal)
            ):
            state[self] = True
            return MATCH, None
        else:
            return FAIL, self

    def expecting(self, state):
        return "(set %s = %s)" % (self.name, self.expectedVal)
# }}}

# {{{ class FunctionCall(SimpleExpectation):
class FunctionCall(SimpleExpectation):

    def __init__(self, name, expectedArgs, expectedKeywords, returnVal):
        self.name = name
        self.expectedArgs = expectedArgs
        self.expectedKeywords = expectedKeywords
        self.returnVal = returnVal

    def test(self, event, state):
        if (isinstance(event, CallFunction) and
            valuesAreExpected(event.args, self.expectedArgs) and
            keywordsAreExpected(event.keywords, self.expectedKeywords)
            ):
            state[self] = True
            return MATCH, self.returnVal
        
        else:
            return FAIL, self

    def expecting(self, state):
        return ("function call: %s(%s) result:%s"
                % (self.name,
                   ', '.join([`x` for x in self.expectedArgs] +
                             ["%s=%s" % (k, `v`)
                              for k,v in self.expectedKeywords.items()]),
                   self.returnVal))
# }}}

# {{{ class Seq(Expectation):
class Seq(Expectation):

    def __init__(self, expectations):
        self.expectations = expectations
    
    def test(self, event, state):
        pos = state.setdefault(self, 0)
        if pos >= len(self.expectations):
            return FAIL, self

        current = self.expectations[pos]
        res, returnVal = current.test(event, state)

        if res == FAIL:
            return FAIL, returnVal
        
        elif res == MATCH:
            if current.complete(state):
                state[self] += 1
            return MATCH, returnVal

        elif res == NOMATCH:
            state[self] += 1
            return self.test(event, state)

    def complete(self, state):
        pos = state.get(self)
        return pos == len(self.expectations)

    def needsMore(self, state):
        pos = state.get(self)
        for e in self.expectations[pos:]:
            if e.needsMore(state):
                return True
        return False

    def expecting(self, state):
        pos = state.get(self)
        if pos < len(self.expectations):
            return "(seq: %s)" % ' '.join([exp.expecting(state)
                                           for exp in self.expectations[pos:]])
        else:
            return "nothing else (end of seq)"
    
    def reset(self, state):
        pos = state.setdefault(self, 0)
        for i in range(pos):
            self.expectations[i].reset(state)
        state[self] = 0
# }}}

# {{{ class Par(Expectation):
class Par(Expectation):

    def __init__(self, expectations):
        self.expectations = expectations
    
    def test(self, event, state):
        done = state.setdefault(self, [])
        for i, e in enumerate(self.expectations):
            if i not in done:
                res, returnVal = e.test(event, state)
                
                if res == MATCH:
                    if e.complete(state):
                        done.append(i)

                    return MATCH, returnVal
                    
        return FAIL, self

    def complete(self, state):
        done = state.get(self) or []
        return len(done) == len(self.expectations)

    def needsMore(self, state):
        return reduce(operator.or_, [e.needsMore(state)
                                     for e in self.expectations])

    def expecting(self, state):
        done = state.get(self) or []
        remaining = [e for i,e in enumerate(self.expectations) if i not in done]
        return "(par: %s)" % ", ".join([x.expecting(state) for x in remaining])

    def reset(self, state):
        for e in self.expectations:
            e.reset(state)
        state[self] = []
# }}}

# {{{ class Choice(Expectation):
class Choice(Expectation):
    def __init__(choices):
        self.choices = choices
    
    def test(self, event):
        chosen = state.setdefault(self, None)
        if chosen == None:
            for e in self.choices:
                res, returnVal = e.test(event, state)

                if res == MATCH:
                    state[self] = e
                    return res, returnVal

            return FAIL, self
        else:
            return chosen.test(event, state)

    def complete(self, state):
        chosen = state[self]
        return chosen != None and chosen.complete(state)

    def needsMore(self, state):
        chosen = state[self]
        if chosen:
            return chosen.needsMore(state)
        else:
            for e in self.choices:
                if e.needsMore():
                    return True
            return False

    def expecting(self, state):
        chosen = state[self]
        if chosen == None:
            return "(choice: %s)" + " ".join([x.expecting(state)
                                              for x in self.content])
        else:
            return chosen.expecting(state)

    def reset(self, state):
        chosen = state[self]
        if chosen != None:
            self.chosen.reset(state)
        state[self] = None
# }}}

# {{{ class Rep(Expectation):
class Rep(Expectation):

    def __init__(self, expectation):
        self.expectation = expectation
    
    def test(self, event, state):
        exp = self.expectation
        res, returnVal = exp.test(event, state)

        if res == MATCH:
            if exp.complete(state):
                exp.reset(state)
            return MATCH, returnVal
        else:
            return NOMATCH, returnVal

    def complete(self, state):
        return False

    def expecting(self, state):
        return "%s*" % self.expectation.expecting(state)

    def needsMore(self, state):
        return False

    def reset(self, state):
        self.expectation.reset(state)
# }}}

# {{{ def methodCallExpectation(name, args, kws, result):
def methodCallExpectation(name, args, kws, result):
    # expect an attribute-get, the result of which is a mock-function
    # that expects to be called once, and returns \result
    return AttrGet(name, MockFunction(FunctionCall(name, args, kws, result)))
# }}}

# {{{ class ObjectExpectation(Expectation):
class ObjectExpectation(Expectation):
    """Expect an event to occur to a specific object"""

    def __init__(self, mob, expectation):
        self.mob = mob
        self.expectation = expectation

    def test(self, event, state):
        if event.mob is self.mob:
            return self.expectation.test(event, state)
        else:
            return FAIL, self

    def expecting(self, state):
        return "%s: %s" % (self.mob, self.expectation.expecting(state))

    def complete(self, state):
        return self.expectation.complete(state)

    def reset(self, state):
        self.expectation.reset(state)

    def needsMore(self, state):
        return not self.complete(state)
# }}}

# {{{ class DebugExpectation(Expectation):
class DebugExpectation(Expectation):

    def __init__(self, expectation):
        self.expectation = expectation
    
    def test(self, event, state):
        res, returnVal = self.expectation.test(event, state)

        if res == MATCH and debug:
            debug()

        return res, returnVal

    def complete(self, state):
        return self.expectation.complete(state)

    def needsMore(self, state):
        return self.expectation.needsMore(state)

    def expecting(self, state):
        return "%s :debug" % self.expectation.expecting(state)

    def reset(self, state):
        self.expectation.reset(state)
# }}}
# }}}

# {{{ Events
# {{{ def GetAttr(mob, name):
class GetAttr:

    def __init__(self, name):
        self.name=name
        
    def __str__(self):
        return "get attribute: " + self.name
# }}}

# {{{ def GetItem(mob, name):
class GetItem:

    def __init__(self, key):
        self.key = key
        
    def __str__(self):
        return "(get item: %s)" % self.key
# }}}

# {{{ def SetAttr(mob, name, value):
class SetAttr:

    def __init__(self, name, value):
        self.name = name
        self.value = value
        
    def __str__(self):
        return "Set attribute: " + self.name + " = " + str(self.value)
# }}}

# {{{ class CallFunction:
class CallFunction:

    def __init__(self, args, keywords):
        self.args = args
        self.keywords = keywords
        
    def __str__(self):
        return "Call function with: %s" % ', '.join(
            [`x` for x in self.args] +
            ["%s=%s" % (k, `v`) for k,v in self.keywords.items()])
    # }}}
# }}}

# {{{ def makemob(name, expecting=None, klass=None, state=None):
def makemob(name, expecting=None, klass=None, state=None):

    if klass:
        assert issubclass(klass, object), \
               "mock object: old-style classes not supported"
    else:
        klass = object

    # {{{ class MockObject:
    class MockObject(klass):

        # {{{ def __init__(self, name, expectation, state):
        def __init__(self, name, expectation, state):
            dict = object.__getattribute__(self, "__dict__")
            dict["__mobdict__"] = attrdict(
                name=name,
                expectation=expectation,
                state=state or {})
        # }}}

        # {{{ def __getattribute__(self, name):
        def __getattribute__(self, name):
            dict = object.__getattribute__(self, "__dict__")
            dictval = dict.get(name)
            if dictval:
                return dictval
            
            fields = self.__mobdict__
            event = GetAttr(name)
            event.mob = self
            res, retval = fields.expectation.test(event, fields.state)
            if res != MATCH:
                if name.startswith("__"):
                    try:
                        return object.__getattribute__(self, name)
                    except AttributeError:
                        pass
                raise ExpectationFailure("%s:\n"
                                         "Expecting: %s\n"
                                         "      Got: %s" %
                                         (fields.name,
                                          fields.expectation.expecting(fields.state),
                                          event))
            else:
                return retval
        # }}}

        # {{{ def __setattr__(self, name, value):
        def __setattr__(self, name, value):
            fields = self.__mobdict__
            if name in self.__dict__:
                self.__dict__[name] = value
            else:
                event = SetAttr(name, value)
                event.mob = self
                res, retval = fields.expectation.test(event, fields.state)
                if res == FAIL:
                    raise ExpectationFailure("%s:\n"
                                             "Expecting: %s\n"
                                             "      Got: %s" %
                                             (fields.name,
                                              retval.expecting(fields.state),
                                              event))
        # }}}

        # {{{ def __getitem__(self, key):
        def __getitem__(self, key):
            fields = self.__mobdict__
            event = GetItem(key)
            event.mob = self
            res, retval = fields.expectation.test(event, fields.state)
            if res != MATCH:
                raise ExpectationFailure("%s:\n"
                                         "Expecting: %s\n"
                                         "      Got: %s" %
                                         (fields.name,
                                          retval.expecting(fields.state),
                                          event))
            else:
                return retval
        # }}}

        # {{{ __repr__ & __str__
        def __repr__(self):
            typ = klass is not object and ":" + klass.__name__ or ""
            return '<mock: %s%s>' % (self.__mobdict__.name, typ)

        __str__ = __repr__
        # }}}
    # }}}

    return MockObject(name, expecting, state)
# }}}

# {{{ def resetmob(mob):
def resetmob(mob):
    mob.__mobdict__.expectation.reset(mob.__mobdict__.state)
# }}}

# {{{ def confirmdone(mob):
def confirmdone(mob):
    fields = mob.__mobdict__
    if fields.expectation.needsMore(fields.state):
        raise ExpectationFailure("Still expecting: " +
                                 fields.expectation.expecting(fields.state))
# }}}

# {{{ class MockFunction:
class MockFunction:

    def __init__(self, expectation):
        self.expectation = expectation

    def __call__(self, *args, **kws):
        event = CallFunction(args, kws)
        res, retval = self.expectation.test(event, {})
        if res == FAIL:
            raise ExpectationFailure("Mock function:\nExpecting %s\nGot %s" %
                                     (retval.expecting({}), event))
        else:
            if type(retval) is types.FunctionType:
                return retval(*args, **kws)
            else:
                return retval
# }}}

# {{{ class MobGroup:
class MobGroup:

    def __init__(self, expectation):
        self.expectation = expectation
        self.state = {}
        
    # {{{ def confirmdone(self):
    def confirmdone(self):
        if self.expectation.needsMore(self.state):
            raise ExpectationFailure("Still expecting: " +
                                     self.expectation.expecting(self.state))
    # }}}
# }}}

# {{{ class TestRunner:
class TestRunner:

    # {{{ def __init__(self, testgroup):
    def __init__(self, testgroup):
        self.topgroup = testgroup
        self.testCount = 0
        self.assertCount = 0
        self.failCount = 0
    # }}}

    # {{{ def currentfunc(self):
    def currentfunc(self):
        return "test_" + self.selector[self.depth]
    # }}}

    # {{{ def currentTestName(self):
    def currentTestName(self):
        return '.'.join(self.selector)
    # }}}

    # {{{ def mayberuntest(self, testfunc):
    def mayberuntest(self, testfunc):
        if testfunc.__name__ == self.currentfunc():
            self.runtest(testfunc)
    # }}}

    # {{{ def maybeNotifyDisabled(self, testfuncname):
    def maybeNotifyDisabled(self, testfuncname):
        if testfuncname == self.currentfunc():
            print self.currentTestName(), "DISABLED"
    # }}}

    # {{{ def runtest(self, testfunc):
    def runtest(self, testfunc):
        self.testCount += 1

        assertsBefore = self.assertCount
        if self.verbose:
            print self.currentTestName(),

        try:
            testfunc()
            if self.verbose:
                print "(%s asserts)" % (self.assertCount - assertsBefore)
        except SkipRestOfTest:
            if self.verbose: print
            pass
        except QuitTests:
            if self.verbose: print
            raise
        except:
            if self.verbose: print
            self.testRaisedException()
    # }}}

    # {{{ def mayberungroup(self, groupfunc):
    def mayberungroup(self, groupfunc):
        if groupfunc.__name__ == self.currentfunc():
            self.rungroup(groupfunc)
    # }}}

    # {{{ def rungroup(self, groupfunc):
    def rungroup(self, groupfunc):
        selector = self.selector
        filter = self.filter
        
        subs = groupfunc.subs

        if len(subs) > 0:
            self.depth += 1
            depth = self.depth

            filtered = depth < len(filter) and filter[depth]
            if filtered and filtered not in subs:
                print 'no such test/group: %s' % filtered
            else:
                if len(selector) == depth:
                    # entering this group for the first time
                    first = filtered and subs.index(filtered) or 0
                    selector.append(subs[first])

                try:
                    self.redo = False
                    groupfunc()
                except QuitTests:
                    raise
                except:
                    self.testRaisedException()

                if self.redo:
                    self.redo = False
                elif len(selector) == depth+1:
                    # group is finished 

                    index = subs.index(selector[depth])
                    if filtered or index == len(subs)-1:
                        selector.pop()
                    else:
                        selector[depth] = subs[index+1]
                    
            self.depth -= 1
    # }}}

    # {{{ def run(self, filter=None, verbose=False)
    def run(self, filter=None, verbose=False, informQuit=False):
        self.filter = filter and filter.split('.') or []
        self.verbose = verbose
        self.current = []
        self.selector = []
        self.depth = -1

        try:
            self.rungroup(self.topgroup)
            while len(self.selector) > 0:
                self.rungroup(self.topgroup)
            userquit = False
        except QuitTests:
            userquit = True
        
        if verbose: print
        print ("Ran %d tests, %d assertions, with %d failures" %
               (self.testCount, self.assertCount, self.failCount))

        if informQuit:
            return userquit
    # }}}

    # {{{ def getSource(self, frame):
    def getSource(self, frame):
        res = ''

        try:
            src, srcLine = inspect.findsource(frame)
            excLine = frame.f_lineno
            srcFile = inspect.getsourcefile(frame)
            res +=  '  [%s]\n' % srcFile
            startLine = max(1, excLine - 6)
            endLine = min(len(src), excLine + 6)
            for line in range(startLine, endLine):
                if line == excLine:
                    arrow = "  ->"
                else:
                    arrow = "    "
                res += arrow + ("%4d: " % line) + src[line-1]
            return res, srcFile, excLine
        except IOError:
            return '[source not available]', None, None
        except:
            return '[source not available (pysco?)]', None, None
    # }}}

    # {{{ def assertPass(self):
    def assertPass(self):
        self.assertCount += 1
    # }}}

    # {{{ def prompt(self, frame, file, line):
    def prompt(self, frame, file, line):
        if file:
            file = file[file.rfind("\\")+1:]
        while 1:
            try:
                choice = raw_input('>> ').strip()
                if dynreload_all: dynreload_all()
            except EOFError:
                raise QuitTests()
            
            if choice == 'd' and frame != None:
                if ipyDebugger:
                    debugger = ipyDebugger.Pdb()
                    debugger.reset()
                    debugger.interaction(frame, None)
                    print self.topgroup.__module__ + ": " + self.currentTestName()
                else:
                    print "IPython debugger not available"

            elif choice == 'de':
                postmortem(self.traceback)
            
            elif choice == 'q':
                raise QuitTests()
            elif choice == 'e':
                emacsto(file, line)
            elif choice == 'c':
                break
            elif choice == 'r':
                self.redo = True
                self.testCount -= 1
                break
            elif choice != '':
                print "Unknown command"
    # }}}

    # {{{ def assertFailure(self, message=None):
    def assertFailure(self, message=None):
        self.testCount += 1
        self.failCount += 1

        thispackage = __name__[:__name__.find(".")+1]
        i = 1
        while 1:
            frame = sys._getframe(i)
            framePackage = frame.f_globals.get('__name__', "")
            if not framePackage.startswith(thispackage):
                break
            i += 1

        src, file, line = self.getSource(frame)

        print "---Assertion Failure---"
        print self.currentTestName()
        print src
        if message:
            print message
            print

        self.prompt(frame, file, line)
        if self.redo:
            raise SkipRestOfTest()
    # }}}

    # {{{ def testRaisedException(self):
    def testRaisedException(self):
        try:
            self.failCount += 1
            exc = sys.exc_info()
            tb = exc[2]
            while tb.tb_next != None and \
                      inspect.getfile(tb.tb_next.tb_frame) != "pymock2.pyX":
                tb = tb.tb_next

            src, file, line = self.getSource(tb.tb_frame)

            print "---Test Failure---"
            print self.currentTestName()
            print
            print src
            print "  %s: %s" % (exc[0].__name__, exc[1])

            self.prompt(tb.tb_frame, file, line)
            
        finally: del tb
    # }}}
# }}}

# {{{ coverage
testmodule = None
def settestmodule(mod):
    global testmodule
    testmodule = mod

    coverage.erase()
    if mod:
        coverage.start()
# }}}

# {{{ def emacsto(file, line):
def emacsto(file, line):
    import os
    file = os.path.basename(file)
    print file,line
    os.popen(r'gnuclientw -e (switch-to-buffer \"%s\")'
             '(goto-line %s)' % (file, line)).close()
    os.popen('gnuclient -x').close()
# }}}

    
