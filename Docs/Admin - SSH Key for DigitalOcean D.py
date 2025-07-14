# Creating SSH Key 
# In VSCode terminal

#this command also works in powershell terminal
bash ssh-keygen -t ed25519 -C "your_email@example.com" -f ~/.ssh/digitalocean_catalyst

#When prompted:
'''
Enter passphrase: (optional but recommended for security)
Confirm passphrase: (repeat if you entered one)
'''

#Confirm the keys exist
bash# Display the public key
cat ~/.ssh/digitalocean_catalyst.pub


#Using keys on Droplet server (Ubuntu)
#add the public key to the authorized keys file

