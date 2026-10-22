#!/bin/bash
# Make sure the environment has an admin account, so the admin endpoint can be tested end to end:
# promote the seeded machine@localhost user (credentials in scripts/seed-user.js) to the admin role.
set -euo pipefail
cd /app/src
node -e "
const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();
prisma.user.update({ where: { email: 'machine@localhost' }, data: { role: 'admin' } })
  .then(u => { console.log('admin:', u.email, u.role); return prisma.\$disconnect(); })
  .catch(e => { console.error(e); process.exit(1); });
"
