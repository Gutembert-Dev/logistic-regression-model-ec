#!/bin/bash
echo "## Pipelines  'Running code style analysis'"

pycodestyle explore_ai_demo --config=scripts/.pycodestyle
results_style=$?

if [ $results_style -ne 0 ]
then
  echo "## Pipelines FAILURE: Code style issues detected"
  exit 1
fi
