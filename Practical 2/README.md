# Smart Appliance Troubleshooting System

## Aim
To design and implement a rule-based expert system using Prolog for troubleshooting common household appliance problems.

## Problem Statement
Design and implement a simple rule-based expert system in Prolog that interactively asks the user a series of yes/no questions about a malfunctioning household appliance and identifies the most likely cause using backward-chaining inference.

## Technologies Used
- Prolog
- SWI-Prolog
- Built-in Prolog predicates: read/1, format/2, writeln/1 and cut (!)

## Features
- Power supply or fuse problem detection
- Faulty power switch detection
- Motor problem detection
- Default message for unknown problems
- Interactive yes/no user input
- Rule-based backward chaining

## How to Run

1. Open SWI-Prolog.
2. Navigate to the folder containing assign2.pl.
3. Load the program:

?- [assign2].

4. Run the program:

?- start.

5. Enter y. or n. when prompted.

## Test Cases

### TC1
Input:
Turn on = y
Power indicator = y

Output:
Power supply or fuse problem.

### TC2
Input:
Turn on = y
Power indicator = n
Power switch not responding = y

Output:
Faulty power switch.

### TC3
Input:
Turn on = n
Turn on = n
Unusual noise = y
Motor not running properly = y

Output:
Motor problem.

### TC4
Input:
Turn on = n
Turn on = n
Unusual noise = n

Output:
Unable to determine the problem. Consult a technician.