#!/bin/bash

echo "# Pipeline 'Running tests with coverage'"

# Get the configuration file
rcfile="scripts/.coveragerc"

# Run tests and code coverage
coverage run --rcfile=${rcfile} --concurrency=multiprocessing -m unittest discover -v
res_test=$?

if [[ ${res_test} -ne 0 ]]
then
  echo "## Pipelines FAILURE: Testing step failed"
  exit 1
fi

coverage combine --rcfile=${rcfile}
res_combine=$?
if [[ ${res_combine} -ne 0 ]]
then
  echo "## Pipelines FAILURE: Coverage combine failed"
  exit 1
fi

coverage report --fail-under=50 --rcfile=${rcfile}
coverage_report=$?
if [[ ${coverage_report} -ne 0 ]]
then
  echo "## Pipelines FAILURE: Coverage percentage failed"
  exit 1
fi
