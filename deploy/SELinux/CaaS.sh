#!/bin/sh -e

DIRNAME=`dirname $0`
cd $DIRNAME
USAGE="$0 [ --update ]"
if [ `id -u` != 0 ]; then
echo 'You must be root to run this script'
exit 1
fi

echo "Building and Loading Policy"
set -x
make -f /usr/share/selinux/devel/Makefile CaaS.pp || exit
/usr/sbin/semodule -i CaaS.pp

# Generate a man page off the installed module
sepolicy manpage -p . -d CaaS_t
# Fixing the file context on /var/services/
/sbin/restorecon -F -R -v /var/services/CaaS/
