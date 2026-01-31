import os
import sys
import platform
import subprocess

def create_mac_shortcut(target_path, desktop_path):
    print(f"🍏 Creating Mac Alias on Desktop...")
    shortcut_path = os.path.join(desktop_path, "SMA2150 Command Center")
    
    # AppleScript to create alias
    scpt = f'''
    tell application "Finder"
        make new alias file at POSIX file "{desktop_path}" to POSIX file "{target_path}"
        set name of result to "SMA2150 Command Center"
    end tell
    '''
    try:
        subprocess.run(['osascript', '-e', scpt], check=True)
        print("✅ Success! Check your Desktop.")
    except Exception as e:
        print(f"❌ Failed to create Mac shortcut: {e}")
        # Fallback to symlink
        try:
            if os.path.exists(shortcut_path):
                os.remove(shortcut_path)
            os.symlink(target_path, shortcut_path)
            print("✅ Created Symlink instead.")
        except Exception as e2:
            print(f"❌ Symlink also failed: {e2}")

def create_windows_shortcut(target_path, desktop_path):
    print(f"🪟 Creating Windows Shortcut...")
    shortcut_path = os.path.join(desktop_path, "SMA2150 Command Center.lnk")
    
    # VBScript to create shortcut
    vbs = f'''
    Set oWS = WScript.CreateObject("WScript.Shell")
    sLinkFile = "{shortcut_path}"
    Set oLink = oWS.CreateShortcut(sLinkFile)
    oLink.TargetPath = "{target_path}"
    oLink.WorkingDirectory = "{os.path.dirname(target_path)}"
    oLink.Description = "Launch SMA2150 Bot"
    oLink.Save
    '''
    
    vbs_path = os.path.join(os.path.dirname(target_path), "create_shortcut.vbs")
    with open(vbs_path, "w") as f:
        f.write(vbs)
        
    try:
        subprocess.run(['cscript', '//Nologo', vbs_path], check=True)
        print("✅ Success! Check your Desktop.")
    except Exception as e:
        print(f"❌ Failed: {e}")
    finally:
        if os.path.exists(vbs_path):
            os.remove(vbs_path)

def main():
    system = platform.system()
    home = os.path.expanduser("~")
    desktop = os.path.join(home, "Desktop")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if system == "Darwin": # Mac
        target = os.path.join(current_dir, "Run_Mac.command")
        create_mac_shortcut(target, desktop)
        
    elif system == "Windows":
        target = os.path.join(current_dir, "Run_Windows.bat")
        create_windows_shortcut(target, desktop)
        
    else:
        print(f"❌ Unsupported OS: {system}")

if __name__ == "__main__":
    main()
