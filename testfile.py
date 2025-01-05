from PackageA import f1      #importing modules from PackageA
from PackageA import f2

from PackageA.SubPackageA import f3     #importing modules from subpackage
from PackageA.SubPackageA import f4

from PackageA.SubPackageB import f5

print(f1.print_something())
print(f2.print_something())
print(f3.print_something())
print(f4.print_something())
print(f5.print_something())