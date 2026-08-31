% SMART APPLIANCE TROUBLESHOOTING SYSTEM

start :-
    writeln('--- SMART APPLIANCE DIAGNOSTIC SYSTEM ---'),
    writeln('Answer with y. or n.'),
    diagnose.

% Rule 1

diagnose :-
    ask('Does the appliance fail to turn on'),
    ask('Is there no power indicator'),
    !,
    writeln('-> RESULT: Power supply or fuse problem.').

% Rule 2

diagnose :-
    ask('Does the appliance fail to turn on'),
    ask('Is the power switch not responding'),
    !,
    writeln('-> RESULT: Faulty power switch.').

% Rule 3

diagnose :-
    ask('Does the appliance make unusual noise'),
    ask('Is the motor not running properly'),
    !,
    writeln('-> RESULT: Motor problem.').

% Default rule

diagnose :-
    writeln('-> RESULT: Unable to determine the problem. Consult a technician.').

% Ask user for a yes/no response

ask(Question) :-
    format('~w? (y/n): ', [Question]),
    read(Response),
    Response == y.