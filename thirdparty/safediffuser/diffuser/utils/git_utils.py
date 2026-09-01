import os
from .optional_deps import optional_import
# Only used to stamp runs with a revision; get_git_rev() already tolerates
# failure and returns None, so a missing gitpython degrades to "no rev".
git = optional_import('gitpython', 'recording the git revision of a run')
import pdb

PROJECT_PATH = os.path.dirname(
    os.path.realpath(os.path.join(__file__, '..', '..')))

def get_repo(path=PROJECT_PATH, search_parent_directories=True):
    repo = git.Repo(
        path, search_parent_directories=search_parent_directories)
    return repo

def get_git_rev(*args, **kwargs):
    try:
        repo = get_repo(*args, **kwargs)
        if repo.head.is_detached:
            git_rev = repo.head.object.name_rev
        else:
            git_rev = repo.active_branch.commit.name_rev
    except:
        git_rev = None

    return git_rev

def git_diff(*args, **kwargs):
    repo = get_repo(*args, **kwargs)
    diff = repo.git.diff()
    return diff

def save_git_diff(savepath, *args, **kwargs):
    diff = git_diff(*args, **kwargs)
    with open(savepath, 'w') as f:
        f.write(diff)

if __name__ == '__main__':

    git_rev = get_git_rev()
    print(git_rev)

    save_git_diff('diff_test.txt')