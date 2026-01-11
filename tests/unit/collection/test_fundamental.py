"""
Unit tests for collection.fundamental module
Tests SEC EDGAR fundamental data collection functionality
"""
import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from quantdl.collection.fundamental import extract_concept, MAPPINGS


class TestExtractConcept:
    """Test extract_concept function"""

    @patch.dict('quantdl.collection.fundamental.MAPPINGS', {'rev': ['us-gaap:Revenues', 'us-gaap:SalesRevenueNet']})
    def test_extract_single_tag_found(self):
        """Test extraction when single tag is found"""
        facts = {
            'us-gaap': {
                'Revenues': {
                    'label': 'Revenues',
                    'description': 'Total revenues',
                    'units': {
                        'USD': [
                            {'val': 1000000, 'fy': 2024, 'fp': 'FY'}
                        ]
                    }
                }
            }
        }

        result = extract_concept(facts, 'rev')

        assert result is not None
        assert result['label'] == 'Revenues'
        assert 'USD' in result['units']
        assert result['units']['USD'][0]['val'] == 1000000

    @patch.dict('quantdl.collection.fundamental.MAPPINGS', {'rev': ['us-gaap:Revenues', 'us-gaap:SalesRevenueNet']})
    def test_extract_multiple_tags_merged(self):
        """Test extraction and merging when multiple tags are found"""
        facts = {
            'us-gaap': {
                'Revenues': {
                    'label': 'Revenues',
                    'description': 'Total revenues',
                    'units': {
                        'USD': [
                            {'val': 1000000, 'fy': 2024, 'fp': 'FY', 'accn': '0001', 'frame': 'CY2024', 'filed': '2025-02-01'}
                        ]
                    }
                },
                'SalesRevenueNet': {
                    'label': 'Sales Revenue Net',
                    'description': 'Net sales revenue',
                    'units': {
                        'USD': [
                            {'val': 2000000, 'fy': 2023, 'fp': 'FY', 'accn': '0002', 'frame': 'CY2023', 'filed': '2024-02-01'}
                        ]
                    }
                }
            }
        }

        result = extract_concept(facts, 'rev')

        assert result is not None
        # Should merge units from both tags
        assert 'USD' in result['units']
        assert len(result['units']['USD']) == 2

    @patch.dict('quantdl.collection.fundamental.MAPPINGS', {'ta': ['us-gaap:Assets']})
    def test_extract_tag_not_found(self):
        """Test extraction when tag is not found"""
        facts = {
            'us-gaap': {
                'Liabilities': {
                    'label': 'Liabilities',
                    'description': 'Total liabilities'
                }
            }
        }

        result = extract_concept(facts, 'ta')

        assert result is None

    @patch.dict('quantdl.collection.fundamental.MAPPINGS', {})
    def test_extract_concept_not_in_mappings(self):
        """Test extraction with concept not in MAPPINGS"""
        facts = {}

        with pytest.raises(KeyError, match="Concept 'unknown' not defined"):
            extract_concept(facts, 'unknown')

    @patch.dict('quantdl.collection.fundamental.MAPPINGS', {'rev': 'us-gaap:Revenues'})  # String instead of list
    def test_extract_invalid_mapping_format(self):
        """Test extraction with invalid mapping format (not a list)"""
        facts = {}

        with pytest.raises(ValueError, match="Invalid mapping format"):
            extract_concept(facts, 'rev')

    @patch.dict('quantdl.collection.fundamental.MAPPINGS', {'rev': ['InvalidTag']})
    def test_extract_tag_without_prefix(self):
        """Test extraction with tag missing prefix"""
        facts = {}

        with pytest.raises(ValueError, match="Tag must include prefix"):
            extract_concept(facts, 'rev')


class TestSECClient:
    """Test SECClient class"""

    @patch('quantdl.collection.fundamental.requests.get')
    def test_fetch_company_facts_success(self, mock_get):
        """Test successful company facts fetch"""
        from quantdl.collection.fundamental import SECClient

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'cik': 320193,
            'entityName': 'Apple Inc.',
            'facts': {
                'us-gaap': {
                    'Assets': {
                        'label': 'Assets',
                        'units': {
                            'USD': [
                                {'val': 1000000, 'fy': 2024, 'fp': 'FY'}
                            ]
                        }
                    }
                }
            }
        }
        mock_get.return_value = mock_response

        client = SECClient()
        result = client.fetch_company_facts('320193')

        assert result['cik'] == 320193
        assert 'facts' in result
        mock_get.assert_called_once()

    @patch('quantdl.collection.fundamental.requests.get')
    def test_fetch_company_facts_cik_padding(self, mock_get):
        """Test that CIK is zero-padded correctly"""
        from quantdl.collection.fundamental import SECClient

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'cik': 320193}
        mock_get.return_value = mock_response

        client = SECClient()
        client.fetch_company_facts('320193')

        # Verify URL contains padded CIK
        call_url = mock_get.call_args[1]['url']
        assert 'CIK0000320193.json' in call_url

    @patch('quantdl.collection.fundamental.requests.get')
    def test_fetch_company_facts_http_error(self, mock_get):
        """Test handling of HTTP error"""
        from quantdl.collection.fundamental import SECClient

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.RequestException("404 Not Found")
        mock_get.return_value = mock_response

        client = SECClient()

        with pytest.raises(requests.RequestException, match="Failed to fetch data"):
            client.fetch_company_facts('999999')

    @patch('quantdl.collection.fundamental.requests.get')
    def test_fetch_company_facts_invalid_json(self, mock_get):
        """Test handling of invalid JSON response"""
        from quantdl.collection.fundamental import SECClient
        import json

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_get.return_value = mock_response

        client = SECClient()

        with pytest.raises(ValueError, match="Invalid JSON response"):
            client.fetch_company_facts('320193')

    def test_sec_client_custom_header(self):
        """Test SECClient with custom header"""
        from quantdl.collection.fundamental import SECClient

        custom_header = {'User-Agent': 'custom@example.com'}
        client = SECClient(header=custom_header)

        assert client.header == custom_header

    def test_sec_client_rate_limiter(self):
        """Test SECClient with rate limiter"""
        from quantdl.collection.fundamental import SECClient

        mock_rate_limiter = Mock()
        client = SECClient(rate_limiter=mock_rate_limiter)

        assert client.rate_limiter == mock_rate_limiter


class TestFundamentalExtractor:
    """Test FundamentalExtractor class"""

    def test_extract_field_success(self):
        """Test successful field extraction"""
        from quantdl.collection.fundamental import FundamentalExtractor

        extractor = FundamentalExtractor()

        facts_response = {
            'facts': {
                'us-gaap': {
                    'Assets': {
                        'label': 'Assets',
                        'units': {
                            'USD': [
                                {'val': 1000000, 'fy': 2024, 'fp': 'FY'}
                            ]
                        }
                    }
                }
            }
        }

        result = extractor.extract_field(facts_response, 'Assets', 'us-gaap', '320193')

        assert len(result) == 1
        assert result[0]['val'] == 1000000

    def test_extract_field_no_facts(self):
        """Test extraction when facts are missing"""
        from quantdl.collection.fundamental import FundamentalExtractor

        extractor = FundamentalExtractor()
        facts_response = {}

        with pytest.raises(KeyError, match="No 'facts' data found"):
            extractor.extract_field(facts_response, 'Assets', 'us-gaap', '320193')

    def test_extract_field_no_fact_type(self):
        """Test extraction when fact type is missing"""
        from quantdl.collection.fundamental import FundamentalExtractor

        extractor = FundamentalExtractor()
        facts_response = {'facts': {}}

        with pytest.raises(KeyError, match="No 'us-gaap' data found"):
            extractor.extract_field(facts_response, 'Assets', 'us-gaap', '320193')

    def test_extract_field_field_not_available(self):
        """Test extraction when specific field is not available"""
        from quantdl.collection.fundamental import FundamentalExtractor

        extractor = FundamentalExtractor()
        facts_response = {
            'facts': {
                'us-gaap': {
                    'Liabilities': {}
                }
            }
        }

        with pytest.raises(KeyError, match="Field 'Assets' not available"):
            extractor.extract_field(facts_response, 'Assets', 'us-gaap', '320193')

    def test_extract_field_no_units(self):
        """Test extraction when units are missing"""
        from quantdl.collection.fundamental import FundamentalExtractor

        extractor = FundamentalExtractor()
        facts_response = {
            'facts': {
                'us-gaap': {
                    'Assets': {
                        'label': 'Assets'
                    }
                }
            }
        }

        with pytest.raises(KeyError, match="No 'units' data found"):
            extractor.extract_field(facts_response, 'Assets', 'us-gaap', '320193')

    def test_extract_field_fallback_to_shares(self):
        """Test that shares unit is used as fallback when USD not available"""
        from quantdl.collection.fundamental import FundamentalExtractor

        extractor = FundamentalExtractor()
        facts_response = {
            'facts': {
                'us-gaap': {
                    'CommonStock': {
                        'label': 'Common Stock',
                        'units': {
                            'shares': [
                                {'val': 1000000, 'fy': 2024, 'fp': 'FY'}
                            ]
                        }
                    }
                }
            }
        }

        result = extractor.extract_field(facts_response, 'CommonStock', 'us-gaap', '320193')

        assert len(result) == 1
        assert result[0]['val'] == 1000000

    def test_extract_field_no_usd_or_shares(self):
        """Test extraction when neither USD nor shares units are available"""
        from quantdl.collection.fundamental import FundamentalExtractor

        extractor = FundamentalExtractor()
        facts_response = {
            'facts': {
                'us-gaap': {
                    'Assets': {
                        'label': 'Assets',
                        'units': {
                            'EUR': [
                                {'val': 1000000, 'fy': 2024, 'fp': 'FY'}
                            ]
                        }
                    }
                }
            }
        }

        with pytest.raises(KeyError, match="Neither USD nor shares units available"):
            extractor.extract_field(facts_response, 'Assets', 'us-gaap', '320193')

    def test_normalize_duration_raw_basic(self):
        """Test basic duration normalization"""
        from quantdl.collection.fundamental import FundamentalExtractor

        extractor = FundamentalExtractor()

        raw_data = [
            {
                'val': 100,
                'start': '2024-01-01',
                'end': '2024-03-31',
                'filed': '2024-04-15',
                'frame': 'CY2024Q1'
            },
            {
                'val': 200,
                'start': '2024-04-01',
                'end': '2024-06-30',
                'filed': '2024-07-15',
                'frame': 'CY2024Q2'
            }
        ]

        result = extractor._normalize_duration_raw(raw_data)

        # Should return quarterly frames as-is
        assert len(result) >= 2

    def test_normalize_duration_raw_empty_data(self):
        """Test normalization with empty data"""
        from quantdl.collection.fundamental import FundamentalExtractor

        extractor = FundamentalExtractor()
        result = extractor._normalize_duration_raw([])

        assert result == []

    def test_normalize_duration_raw_missing_fields(self):
        """Test normalization with missing required fields"""
        from quantdl.collection.fundamental import FundamentalExtractor

        extractor = FundamentalExtractor()

        # Data point missing 'filed' field
        raw_data = [
            {
                'val': 100,
                'start': '2024-01-01',
                'end': '2024-03-31',
                'frame': 'CY2024Q1'
                # 'filed' is missing
            }
        ]

        result = extractor._normalize_duration_raw(raw_data)

        # Should skip incomplete data points
        assert result == []

    def test_normalize_duration_raw_pick_frame_exact(self):
        """Annual frame uses exact Q1/Q2/Q3 to derive Q4."""
        from quantdl.collection.fundamental import FundamentalExtractor

        extractor = FundamentalExtractor()

        raw_data = [
            {
                'val': 100.0,
                'start': '2024-01-01',
                'end': '2024-03-31',
                'filed': '2024-05-01',
                'frame': 'CY2024Q1'
            },
            {
                'val': 200.0,
                'start': '2024-04-01',
                'end': '2024-06-30',
                'filed': '2024-08-01',
                'frame': 'CY2024Q2'
            },
            {
                'val': 300.0,
                'start': '2024-07-01',
                'end': '2024-09-30',
                'filed': '2024-11-01',
                'frame': 'CY2024Q3'
            },
            {
                'val': 1000.0,
                'start': '2024-01-01',
                'end': '2024-12-31',
                'filed': '2025-02-01',
                'frame': 'CY2024'
            }
        ]

        result = extractor._normalize_duration_raw(raw_data)

        q4 = next(dp for dp in result if dp.get("frame") == "CY2024")
        assert q4["val"] == 400.0
        assert q4["start"] == "2024-10-01"

    def test_normalize_duration_raw_pick_frame_fallbacks(self):
        """Annual frame uses Q1I and prefix-matched frames when exact missing."""
        from quantdl.collection.fundamental import FundamentalExtractor

        extractor = FundamentalExtractor()

        raw_data = [
            {
                'val': 110.0,
                'start': '2024-01-01',
                'end': '2024-03-31',
                'filed': '2024-05-01',
                'frame': 'CY2024Q1I'
            },
            {
                'val': 210.0,
                'start': '2024-04-01',
                'end': '2024-06-30',
                'filed': '2024-08-01',
                'frame': 'CY2024Q2'
            },
            {
                'val': 310.0,
                'start': '2024-07-01',
                'end': '2024-09-30',
                'filed': '2024-11-01',
                'frame': 'CY2024Q3'
            },
            {
                'val': 1000.0,
                'start': '2024-01-01',
                'end': '2024-12-31',
                'filed': '2025-02-01',
                'frame': 'CY2024'
            }
        ]

        result = extractor._normalize_duration_raw(raw_data)

        q4 = next(dp for dp in result if dp.get("frame") == "CY2024")
        assert q4["val"] == 370.0
        assert q4["start"] == "2024-10-01"

    def test_normalize_duration_raw_pick_frame_prefix(self):
        """Annual frame falls back to prefix match when only CY...Q1A exists."""
        from quantdl.collection.fundamental import FundamentalExtractor

        extractor = FundamentalExtractor()

        raw_data = [
            {
                'val': 120.0,
                'start': '2024-01-01',
                'end': '2024-03-31',
                'filed': '2024-05-01',
                'frame': 'CY2024Q1A'
            },
            {
                'val': 220.0,
                'start': '2024-04-01',
                'end': '2024-06-30',
                'filed': '2024-08-01',
                'frame': 'CY2024Q2'
            },
            {
                'val': 320.0,
                'start': '2024-07-01',
                'end': '2024-09-30',
                'filed': '2024-11-01',
                'frame': 'CY2024Q3'
            },
            {
                'val': 1000.0,
                'start': '2024-01-01',
                'end': '2024-12-31',
                'filed': '2025-02-01',
                'frame': 'CY2024'
            }
        ]

        result = extractor._normalize_duration_raw(raw_data)

        q4 = next(dp for dp in result if dp.get("frame") == "CY2024")
        assert q4["val"] == 340.0
        assert q4["start"] == "2024-10-01"

    @patch.dict('quantdl.collection.fundamental.MAPPINGS', {'rev': ['us-gaap:Revenues']})
    def test_extract_prefix_not_in_facts(self):
        """Test extraction when prefix doesn't exist in facts"""
        facts = {
            'dei': {
                'EntityName': {}
            }
        }

        result = extract_concept(facts, 'rev')

        assert result is None

    @patch.dict('quantdl.collection.fundamental.MAPPINGS', {'rev': ['us-gaap:Revenues', 'us-gaap:SalesRevenueNet']})
    def test_extract_multiple_units(self):
        """Test extraction with multiple unit types"""
        facts = {
            'us-gaap': {
                'Revenues': {
                    'label': 'Revenues',
                    'description': 'Total revenues',
                    'units': {
                        'USD': [
                            {'val': 1000000, 'fy': 2024, 'fp': 'FY'}
                        ],
                        'shares': [
                            {'val': 5000, 'fy': 2024, 'fp': 'FY'}
                        ]
                    }
                }
            }
        }

        result = extract_concept(facts, 'rev')

        assert result is not None
        assert 'USD' in result['units']
        assert 'shares' in result['units']
        assert result['units']['USD'][0]['val'] == 1000000
        assert result['units']['shares'][0]['val'] == 5000

    @patch.dict('quantdl.collection.fundamental.MAPPINGS', {'rev': ['us-gaap:Revenues', 'custom:CompanyRevenue']})
    def test_extract_custom_prefix(self):
        """Test extraction with custom (company-specific) prefix"""
        facts = {
            'custom': {
                'CompanyRevenue': {
                    'label': 'Company Revenue',
                    'description': 'Custom revenue metric',
                    'units': {
                        'USD': [
                            {'val': 3000000, 'fy': 2024, 'fp': 'FY'}
                        ]
                    }
                }
            }
        }

        result = extract_concept(facts, 'rev')

        assert result is not None
        assert result['label'] == 'Company Revenue'

    @patch.dict('quantdl.collection.fundamental.MAPPINGS', {'rev': ['us-gaap:Revenues']})
    def test_extract_field_without_units(self):
        """Test extraction when field data doesn't have units"""
        facts = {
            'us-gaap': {
                'Revenues': {
                    'label': 'Revenues',
                    'description': 'Total revenues'
                    # Missing 'units' key
                }
            }
        }

        result = extract_concept(facts, 'rev')

        assert result is not None
        assert result['label'] == 'Revenues'
        # Should handle missing units gracefully

    @patch.dict('quantdl.collection.fundamental.MAPPINGS', {'rev': ['us-gaap:Rev1', 'us-gaap:Rev2']})
    def test_extract_merges_from_multiple_fields(self):
        """Test that units are properly merged from multiple fields"""
        facts = {
            'us-gaap': {
                'Rev1': {
                    'label': 'Revenue 1',
                    'description': 'First revenue tag',
                    'units': {
                        'USD': [
                            {'val': 1000000, 'fy': 2024, 'fp': 'Q1', 'accn': '0003', 'frame': 'CY2024Q1', 'filed': '2024-05-01'}
                        ]
                    }
                },
                'Rev2': {
                    'label': 'Revenue 2',
                    'description': 'Second revenue tag',
                    'units': {
                        'USD': [
                            {'val': 2000000, 'fy': 2024, 'fp': 'Q2', 'accn': '0004', 'frame': 'CY2024Q2', 'filed': '2024-08-01'}
                        ]
                    }
                }
            }
        }

        result = extract_concept(facts, 'rev')

        assert result is not None
        assert 'USD' in result['units']
        # Should have merged data from both tags
        assert len(result['units']['USD']) == 2
        values = [item['val'] for item in result['units']['USD']]
        assert 1000000 in values
        assert 2000000 in values

    @patch.dict('quantdl.collection.fundamental.MAPPINGS', {'rev': ['us-gaap:Revenues:Extra']})
    def test_extract_tag_with_multiple_colons(self):
        """Test extraction with tag containing multiple colons"""
        facts = {
            'us-gaap': {
                'Revenues:Extra': {
                    'label': 'Revenues',
                    'description': 'Revenue with extra namespace',
                    'units': {
                        'USD': [
                            {'val': 1000000, 'fy': 2024, 'fp': 'FY'}
                        ]
                    }
                }
            }
        }

        result = extract_concept(facts, 'rev')

        assert result is not None
        assert result['label'] == 'Revenues'


class TestDurationConcepts:
    """Test DURATION_CONCEPTS constant"""

    def test_duration_concepts_exists(self):
        """Test that DURATION_CONCEPTS is defined"""
        from quantdl.collection.fundamental import DURATION_CONCEPTS

        assert isinstance(DURATION_CONCEPTS, set)
        assert len(DURATION_CONCEPTS) > 0

    def test_duration_concepts_contains_expected_values(self):
        """Test that DURATION_CONCEPTS contains expected concepts"""
        from quantdl.collection.fundamental import DURATION_CONCEPTS

        # Check for key duration concepts
        expected_concepts = ['rev', 'net_inc', 'cfo', 'capex']
        for concept in expected_concepts:
            assert concept in DURATION_CONCEPTS


class TestHeaderConfiguration:
    """Test HEADER configuration"""

    @patch.dict('os.environ', {'SEC_USER_AGENT': 'test@example.com'})
    def test_header_from_env(self):
        """Test that HEADER uses environment variable"""
        # Need to reload module to pick up new env var
        import importlib
        import quantdl.collection.fundamental as fundamental_module
        importlib.reload(fundamental_module)

        from quantdl.collection.fundamental import HEADER

        assert 'User-Agent' in HEADER
        assert HEADER['User-Agent'] == 'test@example.com'

    @patch.dict('os.environ', {}, clear=True)
    def test_header_default_fallback(self):
        """Test that HEADER has default fallback"""
        import importlib
        import quantdl.collection.fundamental as fundamental_module
        importlib.reload(fundamental_module)

        from quantdl.collection.fundamental import HEADER

        assert 'User-Agent' in HEADER
        # Should have a fallback value
        assert HEADER['User-Agent'] is not None


class TestParseDatapoints:
    """Test FundamentalExtractor.parse_datapoints method"""

    def test_parse_datapoints_basic(self):
        """Test basic datapoint parsing"""
        from quantdl.collection.fundamental import FundamentalExtractor
        import datetime as dt

        extractor = FundamentalExtractor()

        raw_data = [
            {
                'val': 1000000,
                'start': '2024-01-01',
                'end': '2024-03-31',
                'filed': '2024-04-15',
                'frame': 'CY2024Q1',
                'form': '10-Q',
                'accn': '0001234567-24-000001'
            }
        ]

        result = extractor.parse_datapoints(raw_data)

        assert len(result) == 1
        assert result[0].value == 1000000
        assert result[0].timestamp == dt.date(2024, 4, 15)
        assert result[0].end_date == dt.date(2024, 3, 31)
        assert result[0].start_date == dt.date(2024, 1, 1)
        assert result[0].frame == 'CY2024Q1'
        assert result[0].form == '10-Q'

    def test_parse_datapoints_with_normalize_duration(self):
        """Test parsing with duration normalization"""
        from quantdl.collection.fundamental import FundamentalExtractor

        extractor = FundamentalExtractor()

        raw_data = [
            {
                'val': 100,
                'start': '2024-01-01',
                'end': '2024-03-31',
                'filed': '2024-04-15',
                'frame': 'CY2024Q1',
                'form': '10-Q',
                'accn': '0001'
            }
        ]

        result = extractor.parse_datapoints(raw_data, normalize_duration=True)

        # Should process duration normalization
        assert isinstance(result, list)

    def test_parse_datapoints_with_require_frame(self):
        """Test parsing with require_frame filter"""
        from quantdl.collection.fundamental import FundamentalExtractor

        extractor = FundamentalExtractor()

        raw_data = [
            {
                'val': 100,
                'start': '2024-01-01',
                'end': '2024-03-31',
                'filed': '2024-04-15',
                'frame': 'CY2024Q1',
                'form': '10-Q',
                'accn': '0001'
            },
            {
                'val': 200,
                'start': '2024-01-01',
                'end': '2024-03-31',
                'filed': '2024-04-15',
                'frame': None,  # Missing frame
                'form': '10-Q',
                'accn': '0002'
            }
        ]

        result = extractor.parse_datapoints(raw_data, require_frame=True)

        # Should only include datapoint with frame
        assert len(result) == 1
        assert result[0].value == 100

    def test_parse_datapoints_instant_frame(self):
        """Test is_instant flag for instant frames"""
        from quantdl.collection.fundamental import FundamentalExtractor

        extractor = FundamentalExtractor()

        raw_data = [
            {
                'val': 100,
                'start': '2024-01-01',
                'end': '2024-03-31',
                'filed': '2024-04-15',
                'frame': 'CY2024Q1I',  # Instant frame (has 'I')
                'form': '10-Q',
                'accn': '0001'
            }
        ]

        result = extractor.parse_datapoints(raw_data)

        assert len(result) == 1
        assert result[0].is_instant is True

    def test_parse_datapoints_without_start_date(self):
        """Test parsing datapoint without start date"""
        from quantdl.collection.fundamental import FundamentalExtractor

        extractor = FundamentalExtractor()

        raw_data = [
            {
                'val': 100,
                # 'start' is missing
                'end': '2024-03-31',
                'filed': '2024-04-15',
                'frame': 'CY2024Q1I',
                'form': '10-Q',
                'accn': '0001'
            }
        ]

        result = extractor.parse_datapoints(raw_data)

        assert len(result) == 1
        assert result[0].start_date is None


class TestEDGARDataSource:
    """Test EDGARDataSource class"""

    @patch('quantdl.collection.fundamental.SECClient')
    def test_edgar_init_with_response(self, mock_sec_client):
        """Test EDGARDataSource initialization with provided response"""
        from quantdl.collection.fundamental import EDGARDataSource

        mock_response = {
            'cik': '320193',
            'entityName': 'Apple Inc.',
            'facts': {}
        }

        source = EDGARDataSource(cik='320193', response=mock_response)

        assert source.cik == '320193'
        assert source.response == mock_response
        # Should NOT call SECClient since response was provided
        mock_sec_client.assert_not_called()

    @patch('quantdl.collection.fundamental.SECClient')
    def test_edgar_init_without_response(self, mock_sec_client_class):
        """Test EDGARDataSource initialization without response (fetches data)"""
        from quantdl.collection.fundamental import EDGARDataSource

        mock_client = Mock()
        mock_client.fetch_company_facts.return_value = {
            'cik': '320193',
            'facts': {}
        }
        mock_sec_client_class.return_value = mock_client

        source = EDGARDataSource(cik='320193')

        # Should fetch company facts
        mock_client.fetch_company_facts.assert_called_once_with('320193')

    @patch.dict('quantdl.collection.fundamental.MAPPINGS', {'rev': ['us-gaap:Revenues']})
    def test_edgar_supports_concept(self):
        """Test supports_concept method"""
        from quantdl.collection.fundamental import EDGARDataSource

        source = EDGARDataSource(cik='320193', response={'facts': {}})

        assert source.supports_concept('rev') is True
        assert source.supports_concept('unknown_concept') is False

    @patch.dict('quantdl.collection.fundamental.MAPPINGS', {'rev': ['us-gaap:Revenues']})
    def test_edgar_extract_concept_success(self):
        """Test extract_concept with successful extraction"""
        from quantdl.collection.fundamental import EDGARDataSource

        response = {
            'facts': {
                'us-gaap': {
                    'Revenues': {
                        'label': 'Revenues',
                        'units': {
                            'USD': [
                                {
                                    'val': 1000000,
                                    'start': '2024-01-01',
                                    'end': '2024-03-31',
                                    'filed': '2024-04-15',
                                    'frame': 'CY2024Q1',
                                    'form': '10-Q',
                                    'accn': '0001'
                                }
                            ]
                        }
                    }
                }
            }
        }

        source = EDGARDataSource(cik='320193', response=response)
        result = source.extract_concept('rev')

        assert result is not None
        assert len(result) == 1
        assert result[0].value == 1000000

    @patch.dict('quantdl.collection.fundamental.MAPPINGS', {'rev': ['us-gaap:Revenues']})
    def test_edgar_extract_concept_not_found(self):
        """Test extract_concept when concept not found"""
        from quantdl.collection.fundamental import EDGARDataSource

        response = {
            'facts': {
                'us-gaap': {}
            }
        }

        source = EDGARDataSource(cik='320193', response=response)
        result = source.extract_concept('rev')

        assert result is None

    def test_edgar_get_coverage_period(self):
        """Test get_coverage_period method"""
        from quantdl.collection.fundamental import EDGARDataSource

        source = EDGARDataSource(cik='320193', response={'facts': {}})
        start, end = source.get_coverage_period()

        assert start == "2009-01-01"
        assert end == "2099-12-31"


class TestFundamental:
    """Test Fundamental class"""

    @patch('quantdl.collection.fundamental.SECClient')
    def test_fundamental_init(self, mock_sec_client_class):
        """Test Fundamental initialization"""
        from quantdl.collection.fundamental import Fundamental

        mock_client = Mock()
        mock_client.fetch_company_facts.return_value = {
            'cik': '320193',
            'facts': {}
        }
        mock_sec_client_class.return_value = mock_client

        fund = Fundamental(cik='320193', symbol='AAPL')

        assert fund.cik == '320193'
        assert fund.symbol == 'AAPL'
        assert len(fund._sources) == 1  # EDGAR source

    @patch('quantdl.collection.fundamental.SECClient')
    def test_fundamental_get_concept_data(self, mock_sec_client_class):
        """Test get_concept_data method"""
        from quantdl.collection.fundamental import Fundamental

        mock_response = {
            'facts': {
                'us-gaap': {
                    'Revenues': {
                        'label': 'Revenues',
                        'units': {
                            'USD': [
                                {
                                    'val': 1000000,
                                    'start': '2024-01-01',
                                    'end': '2024-03-31',
                                    'filed': '2024-04-15',
                                    'frame': 'CY2024Q1',
                                    'form': '10-Q',
                                    'accn': '0001'
                                }
                            ]
                        }
                    }
                }
            }
        }

        mock_client = Mock()
        mock_client.fetch_company_facts.return_value = mock_response
        mock_sec_client_class.return_value = mock_client

        fund = Fundamental(cik='320193')
        result = fund.get_concept_data('rev')

        assert result is not None
        assert len(result) == 1
        assert result[0].value == 1000000

    @patch.dict('quantdl.collection.fundamental.MAPPINGS', {'rev': ['us-gaap:Revenues']})
    @patch('quantdl.collection.fundamental.SECClient')
    def test_fundamental_get_concept_data_with_date_filter(self, mock_sec_client_class):
        """Test get_concept_data with date filtering"""
        from quantdl.collection.fundamental import Fundamental

        mock_response = {
            'facts': {
                'us-gaap': {
                    'Revenues': {
                        'label': 'Revenues',
                        'units': {
                            'USD': [
                                {
                                    'val': 1000000,
                                    'start': '2024-01-01',
                                    'end': '2024-03-31',
                                    'filed': '2024-04-15',
                                    'frame': 'CY2024Q1',
                                    'form': '10-Q',
                                    'accn': '0001'
                                },
                                {
                                    'val': 1100000,
                                    'start': '2024-04-01',
                                    'end': '2024-06-30',
                                    'filed': '2024-07-15',
                                    'frame': 'CY2024Q2',
                                    'form': '10-Q',
                                    'accn': '0002'
                                }
                            ]
                        }
                    }
                }
            }
        }

        mock_client = Mock()
        mock_client.fetch_company_facts.return_value = mock_response
        mock_sec_client_class.return_value = mock_client

        fund = Fundamental(cik='320193')
        # Filter to only get Q1 filing
        result = fund.get_concept_data('rev', start_date='2024-01-01', end_date='2024-05-01')

        assert result is not None
        assert len(result) == 1
        assert result[0].value == 1000000

    @patch('quantdl.collection.fundamental.SECClient')
    def test_fundamental_get_concept_data_not_found(self, mock_sec_client_class):
        """Test get_concept_data when concept not found"""
        from quantdl.collection.fundamental import Fundamental

        mock_client = Mock()
        mock_client.fetch_company_facts.return_value = {'facts': {}}
        mock_sec_client_class.return_value = mock_client

        fund = Fundamental(cik='320193')
        result = fund.get_concept_data('unknown_concept')

        assert result is None

    @patch('quantdl.collection.fundamental.SECClient')
    def test_fundamental_rate_limiter_integration(self, mock_sec_client_class):
        """Test that rate limiter is passed to SECClient"""
        from quantdl.collection.fundamental import Fundamental

        mock_rate_limiter = Mock()
        mock_client = Mock()
        mock_client.fetch_company_facts.return_value = {'facts': {}}
        mock_sec_client_class.return_value = mock_client

        fund = Fundamental(cik='320193', rate_limiter=mock_rate_limiter)

        # Verify rate limiter was passed to client
        call_kwargs = mock_sec_client_class.call_args[1]
        assert 'rate_limiter' in call_kwargs
        assert call_kwargs['rate_limiter'] == mock_rate_limiter


class TestSECClientRateLimiter:
    """Test SECClient rate limiter integration"""

    @patch('quantdl.collection.fundamental.requests.get')
    def test_sec_client_rate_limiter_acquire(self, mock_get):
        """Test that rate limiter acquire is called before request"""
        from quantdl.collection.fundamental import SECClient

        mock_rate_limiter = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'cik': 320193}
        mock_get.return_value = mock_response

        client = SECClient(rate_limiter=mock_rate_limiter)
        client.fetch_company_facts('320193')

        # Verify rate limiter was called
        mock_rate_limiter.acquire.assert_called_once()
