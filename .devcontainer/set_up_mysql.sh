#!/bin/bash
# https://askubuntu.com/questions/1261422/how-to-install-mysql-8-0-with-lower-case-table-names-1-on-ubuntu-server-20-04-lt

## Problemn solved MysQL:
## - Case Sensitivity: https://dev.mysql.com/doc/refman/8.4/en/identifier-case-sensitivity.html / https://dev.mysql.com/doc/refman/8.4/en/server-system-variables.html#sysvar_lower_case_table_names
## - Max Allowed Packet: https://dev.mysql.com/doc/refman/8.4/en/packet-too-large.html
## - Strict Mode: https://dev.mysql.com/doc/refman/8.4/en/sql-mode.html#sqlmode_strict_all_tables
## - Blob type: https://dev.mysql.com/doc/refman/8.4/en/blob.html

set -e  # Exit on any error

echo "Setting up MySQL..."

sudo service mysql stop
sudo rm -rf /var/lib/mysql
sudo mkdir /var/lib/mysql    
sudo chown mysql:mysql /var/lib/mysql
sudo chmod 700 /var/lib/mysql

echo 'lower_case_table_names = 1' >> /etc/mysql/mysql.conf.d/mysqld.cnf
echo 'port = 3307' >> /etc/mysql/mysql.conf.d/mysqld.cnf

sudo mysqld --defaults-file=/etc/mysql/my.cnf --initialize --lower_case_table_names=1 --user=mysql --console

sudo service mysql start

# Wait a moment for MySQL to fully start
sleep 5

TEMP_PASSWORD=$(sudo grep 'temporary password' /var/log/mysql/error.log | tail -1 | awk '{print $NF}')
echo "Setting up MySQL with temporary password: ${TEMP_PASSWORD}"

if [ -z "$TEMP_PASSWORD" ]; then
    echo "Error: Could not extract temporary password from MySQL log"
    exit 1
fi

# Change root password
sudo mysql --connect-expired-password -u root -p"${TEMP_PASSWORD}" -e "ALTER USER 'root'@'localhost' IDENTIFIED BY '';"

echo "Root password changed successfully. Configuring database..."

# Use the new password for subsequent commands
mysql -u root -e "SET GLOBAL max_allowed_packet=1073741824;" && \
mysql -u root -e "SET GLOBAL sql_mode=STRICT_ALL_TABLES;" && \
mysql -u root -e "SET GLOBAL max_connections=5000;" && \
mysql -u root -e "CREATE DATABASE IF NOT EXISTS cache;"

# Import SQL files if they exist
if [ -f ".devcontainer/dump_bird_train_pred.sql" ]; then
    echo "Importing dump_bird_train_pred.sql..."
    mysql -u root cache < .devcontainer/dump_bird_train_pred.sql
fi

if [ -f ".devcontainer/dump_bird_train_target.sql" ]; then
    echo "Importing dump_bird_train_target.sql..."
    mysql -u root cache < .devcontainer/dump_bird_train_target.sql
fi

echo "END of MySQL setup"