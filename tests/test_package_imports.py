def test_import_and_seed():
    import mmm_book

    assert mmm_book.RANDOM_SEED == 42
