# When ssh to another node, ~/.bash_profile will be sourced first.  
# If ~/.bashrc exist, source ~/.bashrc
if [ -f ~/.bashrc ]; then
   source ~/.bashrc
fi
