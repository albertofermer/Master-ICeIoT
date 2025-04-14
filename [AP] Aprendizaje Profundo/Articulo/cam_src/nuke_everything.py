import shutil
from pathlib import Path
import signac


def main():
    print('Do you really want to destroy the entire project? [y/N]: ', end='')
    user_input = input()
    if user_input.lower() not in ('y', 'yes'):
        return

    print('Are you really really sure? [y/N]: ', end='')
    user_input = input()
    if user_input.lower() not in ('y', 'yes'):
        return

    print('Type "nuke" to destroy the project: ', end='')
    user_input = input()
    if user_input == 'nuke':
        print('Destroying everything, you asked for it')
        Path('signac.rc').unlink(missing_ok=True)
        Path('signac_project_document.json').unlink(missing_ok=True)
        shutil.rmtree('workspace', ignore_errors=True)

        print('Rebuild project? [Y/n]: ', end='')
        user_input = input()
        if user_input.lower() in ('n', 'no'):
            return
        else:
            signac.init_project('ordinal-cam')
    else:
        print('Aborted')
    

if __name__ == '__main__':
    main()