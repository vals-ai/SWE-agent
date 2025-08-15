#!/usr/bin/env bash

set -euo pipefail

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

function cleanup {
    echo "Stopping react server"
    npx --prefix "${this_dir}/sweagent/frontend" pm2 delete swe-agent
    echo "Stopping Flask server"
    kill "$flask_pid" 2>/dev/null
    echo "Cleanup complete"
}

function print_log {
    echo "Something went wrong. Here's web_api.log:"
    echo "----------"
    cat web_api.log
    echo "----------"
}

trap print_log ERR
python sweagent/api/server.py > web_api.log 2>&1 &
flask_pid=$!

cd "${this_dir}/sweagent/frontend"
npm install
trap cleanup EXIT
npx pm2 start --name swe-agent npm -- start

echo "* If you are running on your own machine, then a browser window "
echo "  should have already opened. If not, wait a few more seconds, then "
echo "  open your browser at http://localhost:3000"
echo "* If you are running in github codespaces, please click the popup "
echo "  that offers to forward port 3000 (not 8000!)."
echo "  Missed it? Find more information at "
echo "  https://swe-agent.com/latest/installation/codespaces#running-the-web-ui"
echo "* Something went wrong? Please check "
echo "  web_api.log for error messages!"
echo "* See here for more information: https://swe-agent.com/latest/usage/web_ui/"

cd ../../

wait
exit $?
