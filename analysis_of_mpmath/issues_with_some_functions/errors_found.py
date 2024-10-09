from mpmath import *

def issue_1():
    """
        Issue with the function findpoly
    """
    print()
    print("First issue: Can't find a reasonable polynomial but does not return None: ")
    mp.dps = 30
    print(findpoly(pi,47))
    print()
    
def issue_2():
    """
        Issue with the identify function
    """
    print()
    mp.dps = 18
    number_to_identify = mpf(47.0)/mpf(5.0)*mpf(pi)
    print("Trying to identify the number: "+str(number_to_identify)+" as a function of sin(1) and sin(2):")
    expression=identify(number_to_identify, ['sin(1)','sin(2)'])
    print(expression)
    print()
    
def last_issue():
    """
        Issue with the function Hurwitz zeta (inspired by https://github.com/mpmath/mpmath/issues)
    """
    print()
    print("Hurwitz zeta evaluated at (1,0) divides by zero and returns infinity: "+str(zeta(1,-2)))
    print("Hurwitz zeta evaluated at (2,0) returns zero division error: ")
    c = input("Do you want to print the error? (s/n) ")
    if (c=='s' or c=='S'):
        print(zeta(2,0))
    print()

def main():
    """
    1) Issue with the findpoly function: can't find a proper polynomial for which the number made up by the first 30 digits of pi is a 0 but does not return None
    2) Issue with the identify function: can't identify a number and gives absurd expression
    Last) Issue woth Hurwitz zeta (definition at: (document page 248))
    """
    issue_1()
    issue_2()
    last_issue()

    


if __name__=="__main__":
    main()
