# Overwrite ~/.bashrc with custom contents
cat > ~/.bashrc << 'EOF'
# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# Customize the PS1 prompt
export PS1="\[\e[32m\]\u@\h:\w\$\[\e[0m\] "

# Other settings
# Set history to append rather than overwrite
shopt -s histappend
# Check window size after each command and update LINES and COLUMNS
shopt -s checkwinsize

# History configuration
HISTSIZE=1000
HISTFILESIZE=2000

# Enable color support for ls and grep
if [ -x /usr/bin/dircolors ]; then
    eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi
EOF
