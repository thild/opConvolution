dnl Warning: This is an automatically generated file, do not edit!
dnl Process this file with autoconf to produce a configure script.
AC_PREREQ([2.54])
AC_INIT([opConvolution], [0.1], [tony_hild@yahoo.com],
             [opConvolution], [http://inf.ufbr.br/])
AM_INIT_AUTOMAKE([foreign -Wall -Werror])
#AM_INIT_AUTOMAKE([subdir-objects])
AM_MAINTAINER_MODE

AC_SUBST(INTI_CFLAGS)
AC_SUBST(INTI_LIBS)

#
# Init Automake subsystem
#
#AM_INIT_AUTOMAKE([opConvolution], [0.1])

# Need the C++ Compiler, Installer and ranlib to strip libraries
#
AC_PROG_CC
AC_PROG_CXX
AC_C_CONST
AC_C_INLINE
AC_TYPE_SIZE_T

#AC_CONFIG_HEADERS([config.h])

dnl pkg-config
AC_PATH_PROG(PKG_CONFIG, pkg-config, no)
if test "x$PKG_CONFIG" = "xno"; then
        AC_MSG_ERROR([You need to install pkg-config])
fi

AC_PROG_INSTALL

AC_ARG_ENABLE(debug,
	AC_HELP_STRING([--enable-debug],
		[Use 'DEBUG' Configuration [default=NO]]),
		enable_debug=yes, enable_debug=no)
AM_CONDITIONAL(ENABLE_DEBUG, test x$enable_debug = xyes)
if test "x$enable_debug" = "xyes" ; then
	CONFIG_REQUESTED="yes"
fi
AC_ARG_ENABLE(release,
	AC_HELP_STRING([--enable-release],
		[Use 'RELEASE' Configuration [default=YES]]),
		enable_release=yes, enable_release=no)
AM_CONDITIONAL(ENABLE_RELEASE, test x$enable_release = xyes)
if test "x$enable_release" = "xyes" ; then
	CONFIG_REQUESTED="yes"
fi
if test -z "$CONFIG_REQUESTED" ; then
	AM_CONDITIONAL(ENABLE_RELEASE, true)
	enable_release=yes
fi


dnl package checks, common for all configs

dnl package checks, per config


AC_CONFIG_FILES([
Makefile
src/Makefile
])

AC_OUTPUT
