# contrib/dblink/Makefile
MODULE_big = ml

OBJS = \
	$(WIN32RES) \
	ml.o
PG_CPPFLAGS = -I$(libpq_srcdir) -ggdb
SHLIB_LINK_INTERNAL = $(libpq) -lcatboostmodel -lc -lm -pthread

EXTENSION = ml
DATA = ml--0.2.sql 
PGFILEDESC = "ml - prediction data in ml model"

REGRESS = ml
REGRESS_OPTS = --dlpath=$(top_builddir)/src/test/regress

ifdef USE_PGXS
PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
else
SHLIB_PREREQS = submake-libpq
subdir = contrib/ml
top_builddir = ../..
include $(top_builddir)/src/Makefile.global
include $(top_srcdir)/contrib/contrib-global.mk
endif
