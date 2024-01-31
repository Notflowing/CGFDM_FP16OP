PROG =	tf_misfits_gof

SRCS =	cwt.f90 fcoolr.f90 main.f90 mod_glob.f90 morlet.f90 \
	tf_misfits_glob.f90 tf_misfits_loc.f90

OBJS =	cwt.o fcoolr.o main.o mod_glob.o morlet.o tf_misfits_glob.o \
	tf_misfits_loc.o

LIBS =	

CC = cc
CFLAGS = -O
FC = f77
FFLAGS = -O
F90 = gfortran
F90FLAGS = -O
LDFLAGS = -s

all: $(PROG)

$(PROG): $(OBJS)
	$(F90) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)

clean:
	rm -f $(PROG) $(OBJS) *.mod

.SUFFIXES: $(SUFFIXES) .f90

.f90.o:
	$(F90) $(F90FLAGS) -c $<

cwt.o: mod_glob.o
main.o: mod_glob.o
morlet.o: mod_glob.o
tf_misfits_glob.o: mod_glob.o
tf_misfits_loc.o: mod_glob.o
