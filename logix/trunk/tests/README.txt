Logix Unit Tests

Logix tests itself! To pull this off, there must be two copies of the
package. To hack on logix, make a duplicate package called devlogix
(you must call it devlogix) and do your hacking there. The stable
logix package can then be used to test your modified code in devlogix.

These tests can be installed as a package anywhere you like
(e.g. devlogix/tests).

The module testall imports all the other modules, so you can run all
the tests with

testall.run()

