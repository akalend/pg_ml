#!/usr/bin/python3

import os
import subprocess

os.system('mkdir -p /tmp/model')
os.system('cp tests/model/* /tmp/model')
os.system('cp tests/data/* /tmp/model')
# os.system('sudo chown -R postgres /tmp/model')

cwd = os.getcwd()

ml_dir = os.path.dirname(cwd) 

postgres_dir = os.path.dirname(os.path.dirname(ml_dir))


env_path="PATH={}:{}/tmp_install/usr/local/pgsql/bin:/usr/bin:/bin".format(ml_dir,postgres_dir)

lib_path="{}/tmp_install/usr/local/pgsql/lib".format(postgres_dir)

cmd='cp {}/postgresql.conf /tmp/model'.format(cwd)

os.system(cmd)


cmd = [ '../../src/test/regress/pg_regress',
		'--temp-instance=./tmp_check',
		'--inputdir=.',
		'--debug',
		'--dlpath=../../src/test/regress',
		'--dbname=contrib_regression',
		'--temp-config=/tmp/model/postgresql.conf',
		'ml',		# ml.sql
		'init',		# init.sql
		'astra',	# astra.sql
		'titanic',	# titanic.sql
		'json'		# check json model 
	  ]

subprocess.run( cmd, env={'PATH': env_path, "LD_LIBRARY_PATH": lib_path}, cwd=cwd)
