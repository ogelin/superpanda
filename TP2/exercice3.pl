course(inf1005C).
course(inf1500).
course(inf1010).
course(log1000).
course(inf1600).
course(log2810).
course(inf2020).
course(log2410).
course(mth1007).
course(log2990).
course(inf2705).
course(inf2205).
course(inf1900).

%Cours prerequis(X) à Y.
isPreReq(inf1005C,inf1010).
isPreReq(inf1005C,log1000).
isPreReq(inf1005C,inf1600).
isPreReq(inf1500,inf1600).
isPreReq(inf1010,inf2010).
isPreReq(inf1010,log2410).
isPreReq(log1000,log2410).
isPreReq(inf2010,inf2705).

%Est-ce que l'on doit avoir les PreReq et les CoReq?
coreqs(log2810,inf2010).
coreqs(mth1007,inf2705).
coreqs(log2990,inf2705).
coreqs(inf1600,inf1900).
coreqs(log1000,inf1900).
coreqs(inf2205,inf1900).

isCoreq(X,Y) :- coreqs(X, Y); coreqs(Y, X).

coursAPrendreComplet(X,L) :- setof(C, arePreOrCoreq(C,X), L).

arePreOrCoreq(A, B) :- isCoreq(A,B).
arePreOrCoreq(A, B) :- isPreReq(A,B).

arePreOrCoreq(A, B) :- 
	A \= B,
	arePreOrCoreq(A, B).

