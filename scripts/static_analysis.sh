#!/bin/bash
echo "## Pipelines  'Running code static analysis'"

pylint explore_ai_demo --rcfile=scripts/.pylintrc -f text --disable=all
results_lint=$?

if [[ ${results_lint} -ne 0 ]]
then
  echo "## Pipelines FAILURE: Code static analysis failed"
  exit 1
fi
